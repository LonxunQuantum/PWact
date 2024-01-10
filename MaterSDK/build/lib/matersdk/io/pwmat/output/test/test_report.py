import unittest

# python3 -m matersdk.io.pwmat.output.test.test_report
from ..report import Report


class ReportTest(unittest.TestCase):
    def test_report_all(self):
        report_path = "/data/home/liuhanyu/hyliu/pwmat_demo/MoS2/scf/band/REPORT"
        out_fermi_path = "/data/home/liuhanyu/hyliu/pwmat_demo/MoS2/scf/band/OUT.FERMI"
        
        #report_path = "/data/home/liuhanyu/hyliu/pwmat_demo/MoS2/scf/nonscf/dos/REPORT"
        #out_fermi_path = "/data/home/liuhanyu/hyliu/pwmat_demo/MoS2/scf/nonscf/dos/OUT.FERMI"
        report = Report(report_path=report_path)
        
        print("\n1. 能带数:", end="\t")
        print( report.get_num_bands() )
        
        print("\n2. kpoints的数目:", end="\t")
        print(report.get_num_kpts())
        
        print("\n3. 得到所有kpoints的本征能量:")
        print(report.get_eigen_energies())
        
        print("\n4. IN.ATOM: ", end="\t")
        print(report.get_in_atom())
        
        print("\n5. self._is_metal:", end="\t")
        print(report._is_metal(out_fermi_path=out_fermi_path))
        
        print("\n6. 材料体系的 cbm 为:", end="\t")
        print(report.get_cbm(out_fermi_path=out_fermi_path))
        
        print("\n7. 材料体系的 vbm 为:", end="\t")
        print(report.get_vbm(out_fermi_path=out_fermi_path))
        
        print("\n8. 带隙:", end="\t")
        print(report.get_bandgap(out_fermi_path=out_fermi_path))
        
        print("\n9. 带隙类型:", end="\t")
        print(report.get_bandgap_type(out_fermi_path=out_fermi_path))
        
        
if __name__ == "__main__":
    unittest.main()