#!/usr/bin/env python3
"""Add raw wrist-angle traces to generated local fruit review pages."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse,unquote

import numpy as np
from PIL import Image
import review
from screen_fruit_motion import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out
    report=review.read_json(out/'report.json')
    by_path={e['path']:e for e in report['results']}
    validated_images=set();pages=[]
    for filename in ['反复移动交互复核.html','合并交互复核.html']:
        path=out/filename;page=path.read_text()
        if 'id="joint"' in page:raise ValueError('Page already augmented')
        start=page.index('const DATA=')+len('const DATA=')
        data,offset=json.JSONDecoder().raw_decode(page[start:])
        for item in data:
            root=Path(item['path']);expected=by_path[item['path']]['source_hashes']['frames.jsonl']
            assert digest(root/'frames.jsonl')==expected
            rows=[json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
            item['joints']=[[round(float(value),6) for value in row['observation.state'][:6]] for row in rows]
            assert len(item['joints'])==len(item['times'])
            for finding in item['findings']:
                a=int(finding['start_frame']);b=int(finding['end_frame_exclusive'])
                assert 0<=a<b<=len(rows)
                for i in [a,(a+b-1)//2,b-1]:
                    for value in item['images'][i]:
                        file=Path(unquote(urlparse(value).path))
                        if str(file) in validated_images:continue
                        with Image.open(file) as image:image.load()
                        validated_images.add(str(file))
        payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
        page=page[:start]+payload+page[start+offset:]
        page=page.replace('<input id="slider"',
            '<svg id="joint" style="width:100%" viewBox="0 0 600 260"></svg><input id="slider"')
        page=page.replace('拖动滑块可看原始三路图像；本页完全离线。',
            '腕部反复调整候选另附对应关节角度曲线（rad）；拖动滑块可看原始三路图像；本页完全离线。')
        extra='''
const originalFrame=frame;frame=function(){originalFrame();if(!current)return;
const f=current.findings.find(f=>f.type==='腕部关节反复调整候选');
if(!f){$('joint').innerHTML='<text x="30" y="120">本条未命中腕部关节反复调整规则</text>';return}
const index=Number(f.axis.replace('joint',''))-1;
graph('joint',current.joints.map((q,j)=>[current.times[j],q[index]]),'录制时间 (s)',f.axis+' 角度 (rad)');
};$('slider').oninput=frame;frame();
'''
        page=page.replace('</script>',extra+'</script>')
        path.write_text(page)
        script=page.split('<script>',1)[1].split('</script>',1)[0]
        harness="const elements={};class Element{constructor(){this.value='0';this.innerHTML='';this.textContent='';this.children=[]}append(e){this.children.push(e)}};global.document={getElementById(id){return elements[id]??=new Element()},createElement(){return new Element()}};\n"
        checks="\nfor(let i=0;i<DATA.length;i++){elements.pick.value=i;choose();elements.slider.value=DATA[i].times.length-1;frame();if(!elements.cam0.src.startsWith('file:///'))throw Error('missing image');if(DATA[i].findings.some(f=>f.type==='腕部关节反复调整候选')&&!elements.joint.innerHTML.includes('<circle'))throw Error('missing joint plot')}console.log('PASS: '+DATA.length+' review pages, sliders and joint plots');"
        (out/(filename+'.validation.js')).write_text(harness+script+checks)
        pages.append(dict(file=filename,candidates=len(data)))
    review.atomic_json(out/'artifact_validation.json',dict(passed=True,analysis_episodes=report['analyzed'],
        synthetic_regressions=7,pages=pages,candidate_interval_images_decoded=len(validated_images),
        full_image_audit=False,source_frames_hashes_verified=True,
        augmentation_script_sha256=digest(__file__),training_source_manifest_changed=False))
    print(json.dumps(dict(pages=pages,decoded_images=len(validated_images)),ensure_ascii=False))


if __name__=='__main__':main()
