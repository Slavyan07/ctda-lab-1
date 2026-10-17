import pytest
from conftest import frame

from loliland_agent.models import ItemAmount, RecipeCandidate, VisibleStack, VisualState
from loliland_agent.recipes.planner import Evidence, RecipeKnowledgeBase, RecipePlanner
from loliland_agent.resources.tracker import ResourceTracker


def amount(item, count=1):
    return ItemAmount(item=item, count=count)


def candidate(output, ingredients, **kwargs):
    return RecipeCandidate(
        output=amount(*output),
        ingredients=tuple(amount(*i) for i in ingredients),
        complete=True,
        confidence=1.0,
        evidence_text="Проверенный рецепт r1",
        **kwargs,
    )


def add(kb, output, ingredients, **kwargs):
    return kb.add(
        candidate(output, ingredients, **kwargs),
        Evidence(kind="user", reference="r1 NEI", excerpt="Проверено человеком", confidence=1.0),
    )


def test_material_batches_shared_stock_and_surplus(memory):
    kb = RecipeKnowledgeBase(memory, "r1")
    add(kb, ("печь", 1), [("плата", 3), ("провод", 2)])
    add(kb, ("плата", 2), [("провод", 3)])
    add(kb, ("провод", 4), [("медь", 2)])
    plan = RecipePlanner(kb).plan(amount("печь", 2), {"плата": 1, "медь": 3}, raw_resources={"медь"})
    assert plan.crafts == {"печь": 2, "плата": 3, "провод": 4}
    assert plan.raw_required == {"медь": 8} and plan.raw_missing == {"медь": 5}
    assert not plan.blockers


def test_machine_technology_and_unknowns_are_explicit(memory):
    kb = RecipeKnowledgeBase(memory, "r1")
    add(kb, ("плата", 1), [("медь", 2)], machine="пресс", technologies=("электричество",))
    add(kb, ("пресс", 1), [("железо", 4)])
    plan = RecipePlanner(kb).plan(amount("плата", 3), {}, raw_resources={"медь", "железо"})
    assert plan.raw_missing == {"медь": 6, "железо": 4} and plan.crafts["пресс"] == 1
    assert plan.machines == ("пресс",) and "Не подтверждена технология: электричество" in plan.blockers
    other = RecipePlanner(RecipeKnowledgeBase(memory, "r2")).plan(amount("плата"), {})
    assert other.blockers and not other.crafts


def test_alternative_recipes_and_cycles(memory):
    kb = RecipeKnowledgeBase(memory, "r1")
    first = add(kb, ("a", 1), [("b", 1)])
    add(kb, ("a", 2), [("c", 1)])
    add(kb, ("b", 1), [("a", 1)])
    planner = RecipePlanner(kb)
    assert "вариант" in planner.plan(amount("a"), {}).blockers[0]
    assert "Цикл" in planner.plan(amount("a"), {}, choices={"a": first.id}).blockers[0]
    with pytest.raises(ValueError):
        planner.plan(amount("a"), {"a": -1})


def test_document_quotes_validated_before_any_write(memory):
    kb = RecipeKnowledgeBase(memory, "r1")
    good = candidate(("a", 1), [("b", 1)])
    bad = good.model_copy(update={"evidence_text": "придумано"})
    with pytest.raises(ValueError):
        kb.ingest_document((good, bad), "Проверенный рецепт r1", "pack-doc")
    assert not kb.all()
    assert kb.ingest_document((good,), "Проверенный рецепт r1", "pack-doc") == 1
    assert kb.all()[0].evidence.kind == "document"


@pytest.mark.parametrize("update", [{"complete": False}, {"confidence": 0.8}, {"ingredients": ()}])
def test_incomplete_recipes_rejected(memory, update):
    kb = RecipeKnowledgeBase(memory, "r1")
    with pytest.raises(ValueError):
        kb.add(
            candidate(("a", 1), [("b", 1)]).model_copy(update=update),
            Evidence(kind="user", reference="r", excerpt="e", confidence=1.0),
        )


def stack(slot, count, item="медь", container="inventory", confidence=1.0):
    return VisibleStack(container=container, slot=slot, count=count, item=item, confidence=confidence)


def seen(*stacks, origin="llm", scene="inventory"):
    return frame().model_copy(
        update={
            "visual": VisualState(scene=scene, confidence=1.0, stacks=stacks),
            "visual_origin": origin,
        }
    )


def test_visible_slots_upsert_unknown_storage_and_unknown_counts(memory):
    tracker = ResourceTracker(memory, "r1")
    observation = seen(
        stack("0", 4),
        stack("1", None),
        stack("2", 100, confidence=0.6),
        stack("0", 50, container="current_container"),
    )
    tracker.ingest(observation)
    tracker.ingest(observation)
    assert tracker.available() == {"медь": 4}
    tracker.ingest(seen(stack("0", 3)))
    assert tracker.available() == {"медь": 3}
    tracker.ingest(seen(stack("0", 10, container="current_container"), scene="storage"), "металлы")
    assert tracker.available() == {"медь": 13}
    assert ResourceTracker(memory, "r2").available() == {}


def test_dirty_refresh_and_cached_images_cannot_resurrect_old_inventory(memory):
    tracker = ResourceTracker(memory, "r1")
    tracker.ingest(seen(stack("0", 4), stack("1", 8)))
    tracker.invalidate()
    tracker.ingest(seen(stack("0", 4), stack("1", 8), origin="cache"))
    assert tracker.available() == {}
    tracker.ingest(seen(stack("0", 1)))
    assert tracker.available() == {"медь": 1}
    assert tracker.available(max_age=-1) == {}


def test_manual_resource_counts_replaced_by_visual_lower_bound(memory):
    tracker = ResourceTracker(memory, "r1")
    tracker.set_facts(
        {
            "base": "x=1 y=64 z=3",
            "storages": {"металлы": "x=2"},
            "raw_resources": ["медь"],
            "resources": {"inventory": {"медь": 20}},
        }
    )
    assert tracker.available() == {"медь": 20} and "resources" not in tracker.facts()
    tracker.ingest(seen(stack("0", 2)))
    assert tracker.available() == {"медь": 2}
    assert tracker.facts()["base"] == "x=1 y=64 z=3"


@pytest.mark.parametrize(
    "facts",
    [
        {"resources": {"inventory": {"медь": -1}}},
        {"resources": {"inventory": {"медь": True}}},
        {"machines": "пресс"},
        {"base": 12},
        {"storages": {"metal": 4}},
        {"invented": []},
    ],
)
def test_bad_user_facts_do_not_change_memory(memory, facts):
    tracker = ResourceTracker(memory, "r1")
    with pytest.raises(ValueError):
        tracker.set_facts(facts)
    assert not tracker.available()


def test_partial_manual_facts_preserve_existing_base_and_machine(memory):
    tracker = ResourceTracker(memory, "r1")
    tracker.set_facts({"base": "x=1", "machines": ["пресс"]})
    tracker.set_facts({"resources": {"inventory": {"медь": 4}}})
    assert tracker.facts()["base"] == "x=1" and tracker.facts()["machines"] == ["пресс"]
    assert tracker.available() == {"медь": 4}


def test_machine_slots_do_not_pollute_named_storage(memory):
    tracker = ResourceTracker(memory, "r1")
    tracker.ingest(seen(stack("0", 10, container="current_container"), scene="machine"), "металлы")
    assert tracker.available() == {}
