import unittest


# python3 -m matersdk.io.pwmat.utils.test.test_lineLocator
from ..lineLocator import LineLocator


class LineLocatorTest(unittest.TestCase):
    def test_locate_all_lines(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/matersdk/tmp_structure_file"
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo1/PWdata/data1/MOVEMENT"
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/MOVEMENT"
        row_idxs_lst = LineLocator.locate_all_lines(file_path=movement_path,
                                                    content="LATTICE")
        print( len(row_idxs_lst) )



if __name__ == "__main__":
    unittest.main()