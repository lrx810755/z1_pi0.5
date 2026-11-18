def pending_cells(counts,quota=20):
 if any(n<0 or n>quota for n in counts.values()):raise ValueError('invalid quota count')
 return sorted((cell for cell,n in counts.items() if n<quota),key=lambda cell:(cell[1],cell[0]))

def reference_index(base,round_number,trial,count,preferred=None):
 if count<=0:raise ValueError('no reference layouts')
 if preferred is not None and not 0<=preferred<count:raise ValueError('invalid preferred layout')
 if preferred is not None:
  if trial==0 and round_number%2==0:return preferred
  return (base+(round_number-1)//2+trial)%count
 return (base+round_number-1+trial)%count

def prioritize_routes(routes,preferred):
 return sorted(routes,key=lambda route:route.get('strategy')!=preferred)
