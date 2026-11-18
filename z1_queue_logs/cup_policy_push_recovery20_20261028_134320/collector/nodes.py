#!/usr/bin/python3
import sys
if __name__=='__main__':
 role=sys.argv.pop(1)
 if role=='assistant':from assistant import main
 elif role=='recorder':from recorder import main
 else:raise ValueError(role)
 main()
