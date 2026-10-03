"""Small NumPy-only contract checks. Not production raycasting or benchmark tests."""
from pathlib import Path
import json
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'PROTOCOL_SPEC.json').read_text())

def schedule(n):
    if n<200: raise ValueError('insufficient planned frames')
    return list(range(0,n,n//200))[:200]

def hit_parameter_z(k,pose,u,v,z):
    q=np.linalg.solve(k,np.array([u,v,1.]))
    d=pose[:3,:3]@q
    hit=pose[:3,3]+z*d
    cam=np.linalg.solve(pose,np.r_[hit,1.])
    return float(cam[2]),d

def face_owner(owners):
    a=np.asarray(owners)
    return int(a[0]) if a[0]>0 and np.all(a==a[0]) else 0

def admit_depth(hit,measured):
    hit,measured=np.asarray(hit),np.asarray(measured)
    return (np.isfinite(hit)&np.isfinite(measured)&(hit>1e-6)&(measured>1e-6)
            &(np.abs(hit-measured)<=np.maximum(.02,.02*measured)))

def bbox(mask):
    y,x=np.nonzero(mask)
    if not len(x): raise ValueError('empty')
    return (int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1))

def views(rows,k):
    ordered=sorted(rows,key=lambda r:(-r['area'],r['frame'],r['hash']))
    result=[]; seen=set()
    for row in ordered:
        if row['frame'] in seen: continue
        seen.add(row['frame']);result.append(row)
        if len(result)==k: break
    return result

def candidates(raw,painted,min_rows=100):
    active=set(painted[painted>0]); result=[]
    for i in np.unique(raw):
        if i<=0 or i in active: continue
        ix=(raw==i)&(painted==0)
        if ix.sum()>=min_rows: result.append(int(i))
    return result

def append(raw,b,labels,recovered):
    own=b.copy(); sem=labels.copy()
    for i,c in recovered.items():
        if i in own[own>0]: raise ValueError('incumbent collision')
        m=(raw==i)&(b==0)
        own[m]=i;sem[m]=c
    return own,sem

def require_complete(rows,order):
    if len(rows)!=len(order) or {r['scene'] for r in rows}!=set(order):
        raise ValueError('partial or duplicate cohort')
    if any(r['status']!='COMPLETE' for r in rows): raise ValueError('blocked')
    return [next(r for r in rows if r['scene']==s) for s in order]

class Contracts(unittest.TestCase):
    def test_01_schedule(self):
        self.assertEqual(schedule(2000),list(range(0,2000,10)))
        self.assertEqual(len(schedule(2599)),200)
        self.assertEqual(schedule(2599)[-1],2388)
        with self.assertRaises(ValueError):schedule(199)
    def test_02_camera_z_not_range(self):
        k=np.array([[200.,0,100.],[0,200.,80.],[0,0,1.]])
        p=np.eye(4);a=.4;p[:3,:3]=[[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]];p[:3,3]=[2,3,4]
        z,d=hit_parameter_z(k,p,160,100,2.)
        self.assertAlmostEqual(z,2.);self.assertGreater(np.linalg.norm(d),1.)
    def test_03_unknown_face_occludes(self):
        front=(1.,face_owner([0,0,0]));back=(2.,face_owner([7,7,7]))
        self.assertEqual(min([front,back])[1],0)
        self.assertEqual(face_owner([7,7,8]),0)
        self.assertEqual(face_owner([7,7,7]),7)
    def test_04_depth_filter(self):
        a=admit_depth([2.,2.,2.,np.inf,2.],[2.01,2.20,0.,2.,np.nan])
        np.testing.assert_array_equal(a,[True,False,False,False,False])
    def test_05_explicit_bbox(self):
        m=np.zeros((12,13),bool);m[3:8,5:10]=1
        b=bbox(m);self.assertEqual(b,(5,3,10,8))
        legacy=(b[0],b[1],b[2]-1,b[3]-1)
        self.assertEqual(legacy,(5,3,9,7))
    def test_06_prefix_and_no_old_request_required(self):
        r=[{'area':100,'frame':20,'hash':'a'}, {'area':200,'frame':10,'hash':'b'},
           {'area':150,'frame':20,'hash':'c'}, {'area':130,'frame':30,'hash':'d'}]
        self.assertEqual(views(r,1),views(r,3)[:1])
        self.assertEqual([x['frame'] for x in views(r,3)],[10,20,30])
        self.assertTrue(all('native_request' not in x for x in views(r,3)))
    def test_07_registry_and_source_minimum(self):
        r=np.repeat([1,2,3],[110,105,99]);b=np.zeros_like(r);b[:100]=1
        self.assertEqual(candidates(r,b),[2])
    def test_08_append_preserves_old(self):
        r=np.array([1,1,2,2,3]);b=np.array([1,1,0,0,0]);s=np.array([5,5,0,0,0])
        o,l=append(r,b,s,{2:9})
        np.testing.assert_array_equal(o,[1,1,2,2,0]);np.testing.assert_array_equal(l,[5,5,9,9,0])
    def test_09_2x2_same_recovery(self):
        r=np.array([1,1,2,2]);b=np.array([1,1,0,0])
        a2=append(r,b,np.array([3,3,0,0]),{2:8})
        a3=append(r,b,np.array([4,4,0,0]),{2:8})
        np.testing.assert_array_equal(a2[0],a3[0]);np.testing.assert_array_equal(a2[1][2:],a3[1][2:])
    def test_10_fc_only_unknown(self):
        b=np.array([1,1,2,2]); old=np.array([3,3,4,4]);f={1:7,2:None}
        result=np.array([0 if f[i] is None else f[i] for i in b])
        np.testing.assert_array_equal(result,[7,7,0,0]);self.assertFalse(np.array_equal(old,result))
    def test_11_complete_pool_and_units(self):
        r=[{'scene':'b','status':'COMPLETE'},{'scene':'a','status':'COMPLETE'}]
        self.assertEqual([x['scene'] for x in require_complete(r,['a','b'])],['a','b'])
        with self.assertRaises(ValueError):require_complete(r[:1],['a','b'])
        with self.assertRaises(ValueError):require_complete([r[0],r[0]],['a','b'])
        self.assertEqual(f'{.123864*100:.2f}','12.39')
    def test_12_method_and_pool_count(self):
        co=SPEC['cohorts'];ms=SPEC['methods']
        self.assertEqual(sum(sum(len(co[c]) for c in m['cohorts']) for m in ms),172)
        self.assertEqual(sum(len(m['cohorts']) for m in ms),14)
        self.assertEqual(len(ms),8)
        self.assertEqual(len({s.split('_')[0] for s in co['scannet_cf18']}),7)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out={'scope':'SYNTHETIC_SPECIFICATION_CONTRACTS_ONLY','tests_run':result.testsRun,
         'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
         'production_map_model_or_benchmark_run':False}
    (ROOT/'REFERENCE_CHECK_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
