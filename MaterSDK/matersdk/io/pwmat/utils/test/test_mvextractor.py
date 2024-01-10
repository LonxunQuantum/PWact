import unittest

from ..mvextractor import MVExtractor


# python3 -m matersdk.io.pwmat.utils.test.test_mvextractor
class MVExtractorTest(unittest.TestCase):
    def test_get_frame_info(self):
        #movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo1/PWdata/data1/MOVEMENT"
        #movement_path = "/data/home/liuhanyu/hyliu/code/mlff/PWmatMLFF_dev/test/SiC/MD/MOVEMENT"
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        
        mvextractor = MVExtractor(movement_path=movement_path)
        ### Step 0.
        print("0. num_frames = ", end="\t")
        print(mvextractor.get_num_frames())
        
        ### Step 1. Print info
        print()
        print("1. info_labels = ", end="\t")
        print(mvextractor.get_info_labels())
        
        ### Step 2. Print Chunk info
        print()
        print("2. Chunk Info:")
        print("2.1. Chunksizes:")
        #print(mvextractor.get_chunksizes())
        print("2.2. Chunkslices:")
        #print(mvextractor.get_chunkslices())
        
        ### Step 3. 
        print()
        print("3. get_frame_str:")
        #print(mvextractor.get_frame_str(fidx=0))
        
        ### Step 4. 
        print()
        print("4. get_frame_info:")
        box, types, coords, etot, fatoms, virial, eatoms = mvextractor.get_frame_info(fidx=0)
        #print(eatoms)
        
        ### Step 5. 
        info = mvextractor.get_frames_info()
        print(info[1])
        
        
    def test_get_frames_info(self):
        pass
        


if __name__ == "__main__":
    unittest.main()
    