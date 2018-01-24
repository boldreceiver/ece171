import sys
import subprocess
import csv
#set some constants for convenience 

compileCommand = "/opt/cuda/bin/nvcc"
compileInclude = "-I/opt/cuda/samples/common/inc/"
compileScaffold =  "scaffold.cu" 
executeCommand = "./a.out"

with open(sys.argv[1]) as testFiles:
   resultsFile = open(sys.argv[2], 'w')
   testsReader = csv.reader(testFiles)
   for row in testsReader:
      fooFile = row[0]
      fooOwner = row[1]
      subprocess.check_output([compileCommand, compileInclude, compileScaffold, fooFile])
      runtime = subprocess.check_output([executeCommand])
      runtime = runtime.decode('ASCII')
      print("Test submitted by", fooOwner, "Execution Time: ",runtime, "ms")
      resultsFile.write( ','.join([fooOwner, runtime, '\n']))
#print(testFiles.read())
testFiles.close()
resultsFile.close()
