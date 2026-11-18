from pathlib import Path
R=Path(__file__).resolve().parent;B=R/'basic'
p=B/'supervise.py';s=p.read_text().replace('z1_tabletop_raw_cup_basic_20hz_200_20261030','z1_cup_basic400_20hz_20261031').replace('8819','8820')
s=s.replace('from fault_latch import check_fault','from fault_latch import check_fault\nsys.path.insert(0,str(R.parent))\nfrom schedule import pending_cells')
s=s.replace('total=200','total=400').replace('n<=10','n<=20').replace("len(reports)==200 and all(x==10", "len(reports)==400 and all(x==20").replace("completed=200", "completed=400").replace("'/200'", "'/400'")
start=s.index('   for row in range(4):\n    for col in range(5):');end=s.index('   assert len(reports)==400',start)
s=s[:start]+'''   stalled=0;round_number=0
   while len(reports)<400:
    round_number+=1;round_before=len(reports)
    for col,row in pending_cells(counts):
     while counts[col,row]<20:
      before=set(DATA.glob('episode_*'));status('collecting',completed=len(reports),current_grid=[col,row],round=round_number,cell_counts={f'{c},{r}':v for (c,r),v in counts.items()})
      runner=ContinuousTrainingRunner(API,'cup','full',1,target_grid=(col,row),max_failures=3,output=output)
      code=runner.run()
      check_fault(R/'sync_fault.json')
      if code in (3,4):
       print('DEFER CELL',col,row,'remaining',20-counts[col,row],flush=True);break
      if code:raise RuntimeError(f'Expert collection exited {code} at cell {col},{row}')
      new=set(DATA.glob('episode_*'))-before
      assert len(new)==1,new
      p=new.pop();report=audit_episode(p);assert report['grid']==[col,row]
      assert report['seed'] not in {x['seed'] for x in reports},'duplicate new dataset seed'
      reports.append(report);counts[col,row]+=1
      save('episode_audits.json',reports)
      print('AUDITED',len(reports),'/400',p.name,flush=True)
    stalled=stalled+1 if len(reports)==round_before else 0
    if stalled>=3:raise RuntimeError('Three full rounds without progress; missing cells retained for diagnosis')
''' + s[end:]
p.write_text(s)
p=B/'collection.launch.py';p.write_text(p.read_text().replace('/home/lp531/z1_queue_logs/cup_basic_20hz_200_20261030',str(B)))
