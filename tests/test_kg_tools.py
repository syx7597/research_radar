"""集合工具边界测试；无需读取图谱或安装模型依赖。"""
import unittest

from ca_agraphrag.kg_tools import KGTools


class IntersectionTests(unittest.TestCase):
    def setUp(self):
        # 集合运算不使用图谱状态，避免测试依赖大型数据文件。
        self.tools = object.__new__(KGTools)

    def test_empty_operand_annihilates_intersection(self):
        for operands in (([], ["sample"]), (["sample"], []),
                         (["sample"], [], ["sample"])):
            with self.subTest(operands=operands):
                self.assertEqual(self.tools.intersect(*operands), [])

    def test_no_operands_and_only_empty_operands(self):
        self.assertEqual(self.tools.intersect(), [])
        self.assertEqual(self.tools.intersect([]), [])
        self.assertEqual(self.tools.intersect([], []), [])

    def test_none_is_an_empty_operand(self):
        self.assertEqual(self.tools.intersect(None, ["sample"]), [])

    def test_nonempty_intersection_is_unique_and_sorted(self):
        self.assertEqual(self.tools.intersect(["b", "a", "a", "c"], ["b", "a"]), ["a", "b"])

    def test_disjoint_operands(self):
        self.assertEqual(self.tools.intersect(["a"], ["b"]), [])

    def test_tool_dispatch_preserves_empty_operand(self):
        self.assertEqual(self.tools.call("intersect", {"sets": [[], ["sample"]]}), [])


if __name__ == "__main__":
    unittest.main()
