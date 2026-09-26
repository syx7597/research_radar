"""Protocol tests with a tiny in-memory engine; no external data/model required."""
import copy
from dataclasses import dataclass
import unittest

from .executor import parse_program, serialize_program, validate_program
from .repair import Repairer


@dataclass
class Value:
    type: str
    value: object
    unit: str = "1"

    def can_compare(self, other):
        return self.type == other.type and (self.type != "quantity" or self.unit == other.unit)

    def isTime(self):
        return self.type in {"year", "date"}

    def contains(self, other):
        return self.type == other.type and self.value == other.value

    def __str__(self):
        return str(self.value)


class TinyEngine:
    def __init__(self):
        self.entities = {
            "A": {"name": "Alpha", "attributes": [
                {"key": "occupation", "value": Value("string", "pilot"), "qualifiers": {}},
                {"key": "retirement date", "value": Value("year", 1990),
                 "qualifiers": {"start time": [Value("year", 1980)]}},
                {"key": "retirement date", "value": Value("year", 2000),
                 "qualifiers": {"end time": [Value("year", 2010)]}}],
                "relations": [{"predicate": "award", "direction": "forward", "object": "B",
                               "qualifiers": {"point in time": [Value("year", 1999)]}},
                              {"predicate": "member of", "direction": "forward", "object": "B",
                               "qualifiers": {"membership start": [Value("year", 1985)]}}]},
            "B": {"name": "Beta", "attributes": [
                {"key": "height", "value": Value("quantity", 2, "metre"), "qualifiers": {}},
                {"key": "birth date", "value": Value("year", 1970),
                 "qualifiers": {"registration date": [Value("year", 1971)]}}], "relations": []}}
        self.concepts = {}
        self.entity_name_to_ids = {"Alpha": ["A"], "Beta": ["B"]}
        self.concept_name_to_ids = {}
        self.key_type = {"occupation": "string", "retirement date": "year", "height": "quantity",
                         "birth date": "year", "start time": "year", "end time": "year",
                         "point in time": "year", "membership start": "year", "registration date": "year"}

    def _parse_key_value(self, key, literal, typ=None):
        typ = typ or self.key_type[key]
        if typ == "quantity":
            number, _, unit = literal.partition(" ")
            return Value(typ, float(number), unit or "1")
        return Value(typ, int(literal) if typ == "year" else literal)

    def Find(self, dependencies, inputs):
        return (list(self.entity_name_to_ids.get(inputs[0], [])), None)

    def FindAll(self, dependencies, inputs):
        return (list(self.entities), None)

    def Relate(self, dependencies, inputs):
        facts = [fact for fact in self.entities[dependencies[0][0][0]]["relations"]
                 if fact["predicate"] == inputs[0] and fact["direction"] == inputs[1]]
        return ([fact["object"] for fact in facts], facts)

    def _filter(self, dependencies, inputs, typ):
        wanted = self._parse_key_value(inputs[0], inputs[1], typ)
        ids, facts = [], []
        for identifier in dependencies[0][0]:
            for fact in self.entities[identifier]["attributes"]:
                if fact["key"] == inputs[0] and fact["value"] == wanted:
                    ids.append(identifier)
                    facts.append(fact)
        return ids, facts

    def FilterStr(self, dependencies, inputs):
        return self._filter(dependencies, inputs, "string")

    def FilterYear(self, dependencies, inputs):
        return self._filter(dependencies, inputs, "year")

    def QFilterYear(self, dependencies, inputs):
        wanted = self._parse_key_value(inputs[0], inputs[1], "year")
        ids, facts = [], []
        for identifier, fact in zip(*dependencies[0]):
            if wanted in fact["qualifiers"].get(inputs[0], []):
                ids.append(identifier)
                facts.append(fact)
        return ids, facts

    def QueryAttr(self, dependencies, inputs):
        for fact in self.entities[dependencies[0][0][0]]["attributes"]:
            if fact["key"] == inputs[0]:
                value = fact["value"]
        return value

    def QueryAttrQualifier(self, dependencies, inputs):
        wanted = self._parse_key_value(inputs[0], inputs[1])
        for fact in self.entities[dependencies[0][0][0]]["attributes"]:
            if fact["key"] == inputs[0] and fact["value"] == wanted:
                values = fact["qualifiers"].get(inputs[2], [])
                if values:
                    return values[0]
        return None

    def What(self, dependencies, inputs):
        return self.entities[dependencies[0][0][0]]["name"]

    def Count(self, dependencies, inputs):
        return len(dependencies[0][0])


class TinyExecutor:
    backend = "baseline"

    def __init__(self):
        self.engine = TinyEngine()

    def execute(self, program):
        try:
            program = validate_program(program)
            memory = []
            for step in program:
                memory.append(getattr(self.engine, step["function"])(
                    [memory[index] for index in step["dependencies"]], step["inputs"]))
            result = memory[-1]
            answers = [] if result is None else [str(result)]
            return {"valid": True, "prediction": answers[0] if answers else "None",
                    "answers": answers, "empty_result": not answers, "error": None}
        except Exception as error:
            return {"valid": False, "prediction": None, "answers": [], "empty_result": False,
                    "error": f"{type(error).__name__}: {error}"}


def row(text, question="When did Alpha retire?", candidates=1):
    return {"id": "example", "question": question,
            "candidates": [{"program_text": text}] * candidates}


def semantic_output(output):
    output = copy.deepcopy(output)
    for candidate in output["candidates"]:
        candidate.pop("execution_seconds", None)
    output["repair_audit"]["cost"] = {key: value for key, value in output["repair_audit"]["cost"].items()
                                      if not key.endswith("seconds")}
    return output


class RepairProtocolTests(unittest.TestCase):
    def setUp(self):
        self.executor = TinyExecutor()

    def test_gold_and_cached_execution_cannot_influence_repairs(self):
        source = row("Find <arg> Alpha <func> QueryAttr <arg> retirement")
        dirty = copy.deepcopy(source)
        dirty.update(answer="fake", program=[], correct=True)
        dirty["candidates"][0].update(answer="fake", gold_program=[], correct=True,
                                       program=[], valid=True, prediction="fake")
        repairer = Repairer(self.executor, mode="local")
        self.assertEqual(semantic_output(repairer.repair_row(source)),
                         semantic_output(repairer.repair_row(dirty)))
        self.assertNotIn("answer", repairer.repair_row(dirty))

    def test_legal_locally_missing_field_is_not_a_trigger(self):
        result = Repairer(self.executor).repair_row(row("Find <arg> Alpha <func> QueryAttr <arg> height"))
        self.assertFalse(result["candidates"][0]["valid"])
        self.assertTrue(result["candidates"][0]["schema_clean"])
        self.assertEqual(result["repair_audit"]["repair_candidates"], 0)
        self.assertEqual(result["repair_audit"]["parents"][0]["triggers"], [])

    def test_legitimate_zero_and_empty_do_not_trigger(self):
        zero = row("Find <arg> Alpha <func> FilterStr <arg> occupation <arg> actor <func> Count")
        result = Repairer(self.executor).repair_row(zero)
        self.assertEqual(result["candidates"][0]["prediction"], "0")
        self.assertEqual(result["repair_audit"]["repair_candidates"], 0)
        empty = row("Find <arg> Alpha <func> QueryAttrQualifier <arg> retirement date <arg> 1990 <arg> end time")
        result = Repairer(self.executor).repair_row(empty)
        self.assertTrue(result["candidates"][0]["empty_result"])
        self.assertEqual(result["repair_audit"]["repair_candidates"], 0)

    def test_roles_and_explicit_function_types_are_separate(self):
        wrong_role = row("Find <arg> Alpha <func> QueryAttr <arg> award")
        result = Repairer(self.executor).repair_row(wrong_role)
        self.assertEqual(result["repair_audit"]["parents"][0]["triggers"][0]["reason"], "globally_illegal_role_field")
        self.assertTrue(all(candidate["edit"]["to"] in {"occupation", "retirement date"}
                            for candidate in result["candidates"][1:]))
        wrong_type = row("Find <arg> Alpha <func> FilterYear <arg> occupation <arg> 1990 <arg> = <func> Count")
        result = Repairer(self.executor).repair_row(wrong_type)
        self.assertEqual(result["candidates"][1]["edit"]["to"], "retirement date")
        self.assertEqual(result["candidates"][1]["trigger"], "explicit_function_type_mismatch")
        self.assertEqual(result["candidates"][1]["program"][1]["inputs"][1:], ["1990", "="])

    def test_qualifier_candidates_stay_on_attached_relation_facts(self):
        source = row("Find <arg> Alpha <func> Relate <arg> award <arg> forward <func> QFilterYear <arg> membership starts <arg> 1999 <arg> = <func> What")
        local = Repairer(self.executor, mode="local").repair_row(source)
        global_ = Repairer(self.executor, mode="global").repair_row(source)
        self.assertEqual([candidate["edit"]["to"] for candidate in local["candidates"][1:]], ["point in time"])
        self.assertIn("membership start", [candidate["edit"]["to"] for candidate in global_["candidates"][1:]])
        self.assertTrue(local["candidates"][1]["context"]["qualifier_fact_binding"])
        self.assertEqual(local["candidates"][1]["prediction"], "Beta")

    def test_attribute_qualifier_binding_includes_original_literal(self):
        source = row("Find <arg> Alpha <func> QueryAttrQualifier <arg> retirement date <arg> 1990 <arg> ends time")
        result = Repairer(self.executor).repair_row(source)
        self.assertEqual([candidate["edit"]["to"] for candidate in result["candidates"][1:]], ["start time"])

    def test_unknown_or_ambiguous_entity_disables_local_context(self):
        for name in ("Missing", "Ambiguous"):
            self.executor.engine.entity_name_to_ids["Ambiguous"] = ["A", "B"]
            source = row(f"Find <arg> {name} <func> QueryAttr <arg> retirement")
            result = Repairer(self.executor).repair_row(source)
            self.assertEqual(result["repair_audit"]["repair_candidates"], 0)
            self.assertEqual(result["repair_audit"]["parents"][0]["rejections"][0]["reason"], "untrusted_local_context")
            self.assertGreater(Repairer(self.executor, mode="global").repair_row(source)["repair_audit"]["repair_candidates"], 0)

    def test_single_edit_budget_and_original_order(self):
        source = row("Find <arg> Alpha <func> QueryAttr <arg> retirement", candidates=6)
        result = Repairer(self.executor, mode="global").repair_row(source)
        self.assertEqual(result["repair_audit"]["original_candidates"], 4)
        self.assertLessEqual(result["repair_audit"]["repair_candidates"], 4)
        self.assertEqual([candidate["original_index"] for candidate in result["candidates"][:4]], list(range(4)))
        original = parse_program(source["candidates"][0]["program_text"])
        for candidate in result["candidates"][4:]:
            changed = copy.deepcopy(candidate["program"])
            edit = candidate["edit"]
            changed[edit["step"]]["inputs"][edit["input"]] = edit["from"]
            self.assertEqual(changed, original)

    def test_remaining_schema_errors_are_not_hidden(self):
        source = row("Find <arg> Alpha <func> FilterYear <arg> birthdate <arg> 1990 <arg> = <func> QueryAttr <arg> occupation typo")
        result = Repairer(self.executor, mode="global").repair_row(source)
        self.assertEqual(len(result["candidates"][0]["schema_issues"]), 2)
        self.assertGreater(result["repair_audit"]["repair_candidates"], 0)
        for candidate in result["candidates"][1:]:
            self.assertFalse(candidate["schema_clean"])
            self.assertEqual(len(candidate["schema_issues"]), 1)

    def test_parse_errors_are_preserved_without_structural_repair(self):
        result = Repairer(self.executor).repair_row(row("Find <arg> Alpha <func> QueryAttr"))
        self.assertEqual(len(result["candidates"]), 1)
        self.assertFalse(result["candidates"][0]["valid"])
        self.assertEqual(result["repair_audit"]["parents"][0]["rejections"][0]["reason"], "unparseable_program")

    def test_prefix_cost_is_separate_from_candidate_execution_budget(self):
        source = row("Find <arg> Alpha <func> QueryAttr <arg> retirement")
        local = Repairer(self.executor).repair_row(source)
        global_ = Repairer(self.executor, mode="global").repair_row(source)
        self.assertEqual(local["repair_audit"]["cost"]["prefix_calls"], 1)
        self.assertEqual(local["repair_audit"]["cost"]["prefix_executed_steps"], 1)
        self.assertEqual(global_["repair_audit"]["cost"]["prefix_calls"], 0)
        self.assertEqual(local["repair_audit"]["cost"]["repair_execution_calls"],
                         local["repair_audit"]["repair_candidates"])
        self.assertGreater(local["repair_audit"]["cost"]["prefix_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
