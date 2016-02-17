import sys;
import math;
x = int(sys.argv[1])
y = int(sys.argv[2])
sys.stdout.write("%d" % (math.factorial(x) / (math.factorial(y) * math.factorial(x - y))))
