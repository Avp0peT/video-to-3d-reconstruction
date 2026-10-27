"""Conditional single-view calibration under explicit centered square-pixel assumptions.

OpenCV 4.13 calib3d: https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html
Source: https://github.com/opencv/opencv/blob/4.13.0/modules/calib3d/src/calibration.cpp
No neighboring intrinsics, native SfM writes, or physical camera acceptance.
"""
from copy import deepcopy
import math

RIGHTS=dict(geometry_accepted=False,camera_pose_established=False,physical_calibration_verified=False,
            identity_verified=False,native_injected=False,new_cameras=0,new_model_points=0)


class _Numeric(Exception):pass


def assess_query_camera(points,width,height,cv,*,budget=lambda:None):
    budget()
    import numpy as np
    from .tracklet_camera_math import assess_projection
    from kozlov_lab.stereo.baseline.radial_branch import propose_central_domain
    def need(ok,reason):
        if not ok:raise _Numeric(reason)
    if type(width) is not int or type(height) is not int or not 1<=width<=16384 or not 1<=height<=16384:raise ValueError('INVALID_IMAGE_DIMENSIONS')
    if not isinstance(points,list) or len(points)>96:raise ValueError('POINTS_BOUND_96')
    seen=set();ordered=[]
    for point in points:
        budget()
        if not isinstance(point,dict) or type(point.get('point_id')) is not str or not point['point_id'] or point['point_id'] in seen:raise ValueError('INVALID_OR_DUPLICATE_POINT_ID')
        seen.add(point['point_id'])
        arrays=[]
        for key,size in [('xyz',3),('raw_xy',2)]:
            value=point.get(key)
            if not isinstance(value,(list,tuple)) or len(value)!=size or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):raise ValueError('INVALID_NONFINITE_OR_BOOLEAN_GEOMETRY')
            arrays.append(np.array(value,dtype=float))
        if not(.5<=arrays[1][0]<=width-.5 and .5<=arrays[1][1]<=height-.5):raise ValueError('RAW_POINT_OUTSIDE_IMAGE')
        ordered.append((point['point_id'],arrays[0],arrays[1]))
    ordered.sort(key=lambda x:x[0]);ids=[p[0] for p in ordered];hold=[i for i in range(len(ids)) if i%4==3];train=[i for i in range(len(ids)) if i%4!=3]
    result=dict(status='BLOCKED',models=[],points=deepcopy(points),ordered_point_ids=ids,train_ids=[ids[i] for i in train],holdout_ids=[ids[i] for i in hold],width=width,height=height,
        assumptions=['principal point fixed at width/2,height/2 in supplied raw-coordinate domain','fx=fy; skew=0; independently estimated focal length','conditional XYZ treated as fixed estimates, not ground truth'],
        split='lexicographic point_id; indices 3,7,... held out; no reselection',
        quantization='calibrateCamera receives float32 XYZ and 2D; original unquantized points retained; evaluation uses original float64 XYZ',
        negative_control_definition='cyclic +1 raw_xy independently within frozen train and holdout; never moves heldout targets into training; differs from old whole-array shift, numerical control scores not directly comparable',
        thresholds=dict(max_error_px=3.,median_error_px=2.,max_scaled_jacobian_condition=1e6,f_over_max_dimension=[.1,10.],max_abs_k=1.,negative_control_min_median_increase_px=1.,negative_control_min_median_ratio=2.),**RIGHTS)
    if len(ids)<12 or len(train)<9 or len(hold)<3:
        result['models']=[dict(model=name,status='BLOCKED',reasons=['INSUFFICIENT_FROZEN_TRAIN_HOLDOUT'],**RIGHTS) for name in ('SIMPLE_PINHOLE','SIMPLE_RADIAL')]
        return result
    criteria=(cv.TERM_CRITERIA_COUNT|cv.TERM_CRITERIA_EPS,60,1e-9)
    baseflags=cv.CALIB_USE_INTRINSIC_GUESS|cv.CALIB_FIX_PRINCIPAL_POINT|cv.CALIB_FIX_ASPECT_RATIO|cv.CALIB_ZERO_TANGENT_DIST|cv.CALIB_FIX_K2|cv.CALIB_FIX_K3
    X=np.array([p[1] for p in ordered]);image=np.array([p[2] for p in ordered]);cx=width/2.;cy=height/2.
    errorclass=getattr(cv,'error',_Numeric)
    def native(fn,*args,**kwargs):
        budget()
        try:answer=fn(*args,**kwargs)
        except errorclass as exc:raise _Numeric('OPENCV: '+str(exc)) from exc
        budget();return answer
    def fit(name,coordinates):
        row=dict(model=name,status='BLOCKED',reasons=[],flags=int(baseflags|(cv.CALIB_FIX_K1 if name=='SIMPLE_PINHOLE' else 0)),criteria=list(criteria),initialization=None,K=None,R=None,t=None,f=None,k=None,train=None,holdout=None,jacobian=None,radial_domain=None,**RIGHTS)
        try:
            budget();need(np.isfinite(X.astype(np.float32)).all() and np.isfinite(coordinates.astype(np.float32)).all(),'FLOAT32_INPUT_OVERFLOW')
            centered=X[train]-X[train].mean(0);sv=np.linalg.svd(centered,compute_uv=False);budget()
            need(sv[0]>0 and sv[-1]>sv[0]*1e-8,'DEGENERATE_NONPLANAR_TRAINING_SUPPORT')
            training=[dict(point_id=ids[i],xyz=X[i].tolist(),raw_xy=coordinates[i].tolist()) for i in train]
            init=assess_projection(training,budget=budget)['all_fit'];need(init is not None,'TRAINING_DLT_INITIALIZATION_REFUSED')
            row['initialization']=dict(training_point_ids=[ids[i] for i in train],projective_fit=init)
            decomposed=native(cv.decomposeProjectionMatrix,np.array(init['P'],float));kin=np.asarray(decomposed[0],float)
            need(kin.shape==(3,3) and np.isfinite(kin).all() and abs(kin[2,2])>1e-12,'INITIAL_K_INVALID')
            kin=kin/kin[2,2];f0=float((abs(kin[0,0])+abs(kin[1,1]))/2)
            need(math.isfinite(f0) and f0>0,'INITIAL_F_INVALID')
            K0=np.array([[f0,0,cx],[0,f0,cy],[0,0,1]],float);row['initialization'].update(f=f0,K=K0.tolist())
            xyz32=X[train].astype(np.float32).reshape(-1,1,3);xy32=coordinates[train].astype(np.float32).reshape(-1,1,2)
            row['quantized_training_xyz']=xyz32.reshape(-1,3).astype(float).tolist();row['quantized_training_raw_xy']=xy32.reshape(-1,2).astype(float).tolist()
            rms,K,dist,rvecs,tvecs=native(cv.calibrateCamera,[xyz32],[xy32],(width,height),K0.copy(),np.zeros(5,float),flags=row['flags'],criteria=criteria)
            K=np.asarray(K,float);d=np.asarray(dist,float).reshape(-1);rv=np.asarray(rvecs[0],float).reshape(3);t=np.asarray(tvecs[0],float).reshape(3)
            need(math.isfinite(float(rms)) and K.shape==(3,3) and np.isfinite(K).all() and len(d)>=5 and np.isfinite(d).all() and np.isfinite(rv).all() and np.isfinite(t).all(),'NONFINITE_NATIVE_FIT')
            R=np.asarray(native(cv.Rodrigues,rv)[0],float);need(R.shape==(3,3) and np.isfinite(R).all() and np.allclose(R@R.T,np.eye(3),atol=1e-8,rtol=0) and abs(np.linalg.det(R)-1)<1e-8,'INVALID_SO3')
            f=float(K[0,0]);k=float(d[0]);row.update(K=K.tolist(),R=R.tolist(),t=t.tolist(),rvec=rv.tolist(),f=f,k=k,opencv_rms=float(rms),distortion=d.tolist())
            need(np.allclose(K,np.array([[f,0,cx],[0,f,cy],[0,0,1]]),atol=1e-7,rtol=0),'FIXED_INTRINSIC_ASSUMPTION_VIOLATION')
            need(np.max(np.abs(d[1:]))<=1e-10 and (name!='SIMPLE_PINHOLE' or abs(k)<=1e-10),'FIXED_DISTORTION_ASSUMPTION_VIOLATION')
            row['status']='REJECTED'
            if not .1<=f/max(width,height)<=10:row['reasons'].append('FOCAL_LENGTH_BOUND')
            if abs(k)>1:row['reasons'].append('RADIAL_COEFFICIENT_BOUND')
            need(f>0,'NONPOSITIVE_FOCAL_LENGTH')
            try:domain=propose_central_domain([f,cx,cy,k],width,height,coordinates.tolist())
            except (ValueError,ArithmeticError) as exc:raise _Numeric('RADIAL_DOMAIN: '+str(exc)) from exc
            row['radial_domain']=domain
            if not domain['full_image_coverage']['covered'] or not domain['all_observations_invertible']:row['reasons'].append('RADIAL_IMAGE_DOMAIN_NOT_INVERTIBLE')
            projected,jac=native(cv.projectPoints,X.reshape(-1,1,3),rv,t,K,d)
            projected=np.asarray(projected,float).reshape(-1,2);jac=np.asarray(jac,float)
            depths=(X@R.T+t)[:,2];errors=np.linalg.norm(projected-coordinates,axis=1)
            need(np.isfinite(projected).all() and np.isfinite(errors).all() and np.isfinite(depths).all(),'NONFINITE_PROJECTION')
            for label,indices in [('train',train),('holdout',hold)]:
                vals=errors[indices]
                row[label]=dict(count=len(indices),rows=[dict(point_id=ids[i],measured_raw_xy=coordinates[i].tolist(),predicted_raw_xy=projected[i].tolist(),error_px=float(errors[i]),depth=float(depths[i])) for i in indices],max_px=float(vals.max()),median_px=float(np.median(vals)))
                if vals.max()>3 or np.median(vals)>2:row['reasons'].append(label.upper()+'_REPROJECTION_GATE')
            if not np.all(depths>0):row['reasons'].append('NONPOSITIVE_DEPTH')
            # OpenCV Jacobian columns: rvec3,tvec3,fx,fy,cx,cy,k1,... .
            # Shared focal derivative is d/dfx+d/dfy. Normalize each column's
            # L2 norm to remove units; rank remains a separate admission gate.
            jr=jac.reshape(len(X),2,-1)[train].reshape(-1,jac.shape[1]);need(jr.shape[1]>=11 and np.isfinite(jr).all(),'INVALID_TRAINING_JACOBIAN')
            columns=[jr[:,i] for i in range(6)]+[jr[:,6]+jr[:,7]]
            if name=='SIMPLE_RADIAL':columns.append(jr[:,10])
            J=np.column_stack(columns);scales=np.linalg.norm(J,axis=0);need(np.all(scales>0),'ZERO_JACOBIAN_COLUMN')
            budget();singular=np.linalg.svd(J/scales,compute_uv=False);budget()
            rank=int(np.sum(singular>singular[0]*1e-10));condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
            if condition is not None and not math.isfinite(condition):condition=None
            row['jacobian']=dict(parameter_order=['rx','ry','rz','tx','ty','tz','f']+(['k1'] if name=='SIMPLE_RADIAL' else []),column_l2_scales=scales.tolist(),scaled_singular_values=singular.tolist(),rank=rank,condition=condition,scaling='training-only unit column L2 norm')
            if rank!=len(columns) or condition is None or condition>1e6:row['reasons'].append('TRAINING_JACOBIAN_UNIDENTIFIABLE')
            if not row['reasons']:row['status']='CONDITIONAL_CAMERA_CANDIDATE'
        except _Numeric as exc:
            row['reasons'].append(str(exc));row['status']='REJECTED' if row['K'] is not None else 'BLOCKED'
        except np.linalg.LinAlgError as exc:
            row['reasons'].append('NUMERICAL_LINEAR_ALGEBRA: '+str(exc));row['status']='BLOCKED'
        return row
    control=image.copy();control[train]=np.roll(image[train],1,axis=0);control[hold]=np.roll(image[hold],1,axis=0)
    for name in ('SIMPLE_PINHOLE','SIMPLE_RADIAL'):
        budget();row=fit(name,image);budget();negative=fit(name,control);budget()
        row['negative_control']=negative
        row['negative_control']['target_source_point_ids']=dict(train=[ids[train[(j-1)%len(train)]] for j in range(len(train))],holdout=[ids[hold[(j-1)%len(hold)]] for j in range(len(hold))])
        if row['status']=='CONDITIONAL_CAMERA_CANDIDATE':
            # A numerical/native failure is missing control evidence, not proof
            # that the deliberately incorrect correspondence was rejected.
            completed=negative['train'] is not None and negative['holdout'] is not None and negative['jacobian'] is not None
            if not completed:
                row['status']='REJECTED';row['reasons'].append('NEGATIVE_CONTROL_NOT_EVALUATED')
            else:
                documented={'TRAIN_REPROJECTION_GATE','HOLDOUT_REPROJECTION_GATE','NONPOSITIVE_DEPTH','FOCAL_LENGTH_BOUND','RADIAL_COEFFICIENT_BOUND','RADIAL_IMAGE_DOMAIN_NOT_INVERTIBLE','TRAINING_JACOBIAN_UNIDENTIFIABLE'}
                unexpected=[reason for reason in negative['reasons'] if reason not in documented]
                separated=not unexpected and negative['status']=='REJECTED' and bool(negative['reasons'])
                correct=row['holdout']['median_px'];wrong=negative['holdout']['median_px']
                separated=separated or (not unexpected and wrong>=correct+1 and wrong>=2*correct)
                if unexpected:
                    row['status']='REJECTED';row['reasons'].append('NEGATIVE_CONTROL_NOT_EVALUATED')
                elif not separated:
                    row['status']='REJECTED';row['reasons'].append('NEGATIVE_CONTROL_NOT_SEPARATED')
        def finite_json(value):
            if isinstance(value,float) and not math.isfinite(value):
                return None
            if isinstance(value,dict):return {key:finite_json(v) for key,v in value.items()}
            if isinstance(value,list):return [finite_json(v) for v in value]
            return value
        clean=finite_json(row)
        if clean!=row:
            clean['status']='REJECTED';clean['reasons'].append('NONFINITE_EVIDENCE_REPLACED_WITH_NULL')
        result['models'].append(clean)
    result['status']='CONDITIONAL_CAMERA_CANDIDATE' if any(r['status']=='CONDITIONAL_CAMERA_CANDIDATE' for r in result['models']) else 'REJECTED' if any(r['status']=='REJECTED' for r in result['models']) else 'BLOCKED'
    return result
