import unittest,datetime,pickle
import numpy as np
import os.path as op
from src.decomposition.fpmc import FPMC 


class FPMC_Test(unittest.TestCase):

	def test(self):
		data = pickle.load(open(op.join('..','..','output','fpmc_data','training','data'), 'r'))
		fpmc = FPMC()
		(vui,viu,vil,vli) = fpmc.factorize(data, 50, 50, .2, .5)
		
		out_path = op.join('..','..','output','fpmc_data','training')
		np.savetxt(op.join(out_path,'vui'), vui)
		np.savetxt(op.join(out_path,'viu'), viu)
		np.savetxt(op.join(out_path,'vil'), vil)
		np.savetxt(op.join(out_path,'vli'), vli)
		print 'start cal a'
		fpmc.cal_a()
		print 'end cal a'
		pickle.dump(fpmc.a, open(op.join(out_path,'a'), 'w'))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

		