import unittest

# python3 -m matersdk.io.pwmat.input.test.test_gen_vext
from ..gen_vext import GenVext

class GenVextTest(unittest.TestCase):
    def test_all(self):
        ### Step 1.
        vr_center = [0.1, 0.2, 0.3]
        vr_type = 2
        add_vr = True
        a4 = 0.4
        a5 = 0.5
        a6 = 0.6
        a7 = 0.77
        a8 = 0.8
        a9 = 0.9
        gen_vext = GenVext(
                        vr_center,
                        vr_type,
                        add_vr,
                        a4, a5, a6, a7, a8, a9
                        )
        output_path = "/data/home/liuhanyu/hyliu/code/matersdk/matersdk/io/pwmat/input/test/gen.vext"
        gen_vext.to(output_path=output_path)
    
    
if __name__ == "__main__":
    unittest.main()