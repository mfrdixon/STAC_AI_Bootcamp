"""Explicit application-level semantics; never infer a set from observed values."""
import math

def compile_input(payload):
    kind=payload.get('type')
    if kind not in ('set','sequence'):
        raise ValueError('Declare type=set or type=sequence explicitly')
    items=payload.get('items')
    if not isinstance(items,list) or not items:
        raise ValueError('Provide a nonempty items list')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in items):
        raise ValueError('Items must be finite real numbers')
    if kind=='set':
        if payload.get('duplicates','preserve')!='preserve':
            raise ValueError('This sample is a multiset; preserve duplicate observations')
        return {'items':list(items),'block_ids':list(range(len(items))),
                'position_ids':[0]*len(items),'pooling':'mean','mode':'block_set'}
    return {'items':list(items),'block_ids':[0]*len(items),
            'position_ids':list(range(len(items))),'pooling':'mean','mode':'ordinary'}

if __name__=='__main__':
    a=compile_input({'type':'set','items':[.12,-.83,.12]})
    assert a['items']==[.12,-.83,.12] and a['position_ids']==[0,0,0]
    b=compile_input({'type':'sequence','items':[.12,-.83,.12]})
    assert b['block_ids']==[0,0,0] and b['position_ids']==[0,1,2]
    try: compile_input({'items':[1,2]})
    except ValueError: pass
    else: raise AssertionError('Undeclared semantics must be rejected')
    print('Explicit set/sequence semantics, duplicate preservation and missing-type checks passed.')
