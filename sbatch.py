import argparse
import os

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('-J', required=True, help='what job name')
  parser.add_argument('-C', default='"titanx|txpascal"', help='what configureation')
  parser.add_argument('-sh', required=True, help='which bash file to run')
  parser.add_argument('-T', default=1, type=int, help='how many consecutive times to run')
  parser.add_argument('-w', default="", help="specific node")

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  for i in range(params['T']):
    cmd = "sbatch -p greg-gpu -C %s -J %s -d singleton -o %s_%s.out " %(params['C'], params['J'], params['J'], i)
    if params['w']!="":
      cmd = cmd + "-w " + params['w'] + ' '
    cmd = cmd + params['sh']
    print(cmd)
    os.system(cmd)

