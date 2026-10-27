"""Persist bounded batch events before publishing candidate artifacts."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path


class BatchEvidence:
    """One frozen plan, sequential batches, immutable event files, visible partial state."""
    def __init__(self, root):
        self.root=Path(root)
        if self.root.exists():raise ValueError('fresh batch evidence directory required')
        self.plan=None;self.states=[];self.events=[];self._bindings={}

    def _save(self,path,value,*,replace=False):
        if path.exists() and not replace:raise ValueError('batch evidence event already exists')
        content=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('x',encoding='utf-8',newline='\n') as f:f.write(content)
        os.replace(temporary,path)
        data=content.encode('utf-8')
        self._bindings[path.name]=dict(path=str(path.resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

    def __call__(self,event):
        value=deepcopy(event);kind=value.get('kind')
        if kind=='PLAN':
            if self.plan is not None:raise ValueError('batch plan already frozen')
            batches=value['batches'];profile=value['profile']
            if (not isinstance(batches,list) or len(batches)>3
                    or profile.get('batch_size')!=32 or profile.get('max_batches')!=3
                    or profile.get('max_groups')!=96):raise ValueError('invalid batch plan budget')
            flattened=[]
            for i,b in enumerate(batches):
                if (type(b.get('batch_id')) is not int or b['batch_id']!=i
                        or b.get('status')!='NOT_STARTED' or not isinstance(b.get('group_ids'),list)
                        or not 1<=len(b['group_ids'])<=32):raise ValueError('invalid batch partition')
                flattened.extend(b['group_ids'])
            if (not all(type(g) is str and g for g in flattened) or len(set(flattened))!=len(flattened)
                    or flattened!=value['selected_groups']):raise ValueError('batch plan selected groups mismatch')
            self.root.mkdir(parents=True,exist_ok=False)
            filename='PLAN.json';states=[dict(batch_id=b['batch_id'],group_ids=b['group_ids'],status='NOT_STARTED') for b in batches]
        else:
            if self.plan is None:raise ValueError('batch event before frozen plan')
            i=value.get('batch_id')
            if type(i) is not int or not 0<=i<len(self.states):raise ValueError('batch event identity')
            states=deepcopy(self.states);current=states[i]
            if value.get('group_ids')!=current['group_ids']:raise ValueError('batch event group mismatch')
            if any(s['status']!='COMPLETE' for s in states[:i]) or any(s['status']!='NOT_STARTED' for s in states[i+1:]):
                raise ValueError('batch event out of order')
            transition={'BATCH_BEGIN':('NOT_STARTED','RUNNING','begin'),
                'BATCH_END':('RUNNING','COMPLETE','end'),'BATCH_FAILED':('RUNNING','FAILED','failed')}.get(kind)
            if transition is None or current['status']!=transition[0] or value.get('status')!=transition[1]:
                raise ValueError('invalid batch state transition')
            current['status']=transition[1]
            if kind=='BATCH_FAILED':current['failure_type']=value.get('failure_type',value.get('reason'))
            filename=f'batch_{i:03d}_{transition[2]}.json'
        self._save(self.root/filename,value)
        if kind=='PLAN':self.plan=value
        self.states=states;self.events.append(filename)
        self._save(self.root/'BATCH_STATUS.json',self.summary(),replace=True)

    def summary(self):
        complete=sum(s['status']=='COMPLETE' for s in self.states)
        status=('NOT_PLANNED' if self.plan is None else 'FAILED' if any(s['status']=='FAILED' for s in self.states)
                else 'COMPLETE' if complete==len(self.states) else 'INCOMPLETE')
        return dict(status=status,planned_batches=len(self.states),completed_batches=complete,
            batches=deepcopy(self.states),events=list(self.events),
            evidence_scope='Completed batch evidence only; partial state never grants final geometry acceptance')

    def bindings(self):
        if self.summary()['status']!='COMPLETE':raise ValueError('incomplete batches cannot publish candidates')
        result=[]
        for name in self.events+['BATCH_STATUS.json']:
            path=(self.root/name).resolve(strict=True);data=path.read_bytes()
            bound=dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            if bound!=self._bindings[name]:raise ValueError('batch evidence changed before publication')
            result.append(bound)
        return result
