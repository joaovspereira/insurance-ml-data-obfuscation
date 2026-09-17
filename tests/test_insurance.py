import unittest
import numpy as np
import pandas as pd
from insurance import demo,fit_ols,predict_ols,verify_invariance,evaluate_original

class LinearAlgebraTests(unittest.TestCase):
    def test_nonorthogonal_transformation_preserves_predictions(self):
        r=demo()
        self.assertTrue(r['predictions_equal_at_1e_8'])
        self.assertLess(r['prediction_max_absolute_difference'],1e-10)
        self.assertLess(r['recovery_max_absolute_difference'],1e-10)
    def test_intercept_and_known_linear_model(self):
        X=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=float)
        y=5+X@np.array([2,3])
        np.testing.assert_allclose(fit_ols(X,y),[5,2,3],atol=1e-12)
        np.testing.assert_allclose(predict_ols(X,fit_ols(X,y)),y)
    def test_singular_transform_and_rank_deficiency_rejected(self):
        X=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=float)
        with self.assertRaises(ValueError): verify_invariance(X,X,np.arange(4),np.zeros((2,2)))
        with self.assertRaises(ValueError): verify_invariance(np.ones((4,2)),X,np.arange(4),np.eye(2))
    def test_original_workflow_on_synthetic_schema(self):
        rng=np.random.default_rng(5); n=160
        df=pd.DataFrame({'Gender':rng.integers(0,2,n),'Age':rng.integers(18,70,n),
            'Salary':rng.uniform(20000,90000,n),'Family members':rng.integers(0,5,n)})
        df['Insurance benefits']=(df['Age']>45).astype(int)
        r=evaluate_original(df)
        self.assertEqual(r['train_rows']+r['test_rows'],n)
        self.assertTrue(np.isfinite(r['test_r2']))
        self.assertTrue(r['transformation_check']['predictions_equal_at_1e_8'])
