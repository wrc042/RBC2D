python .\exec\rbc2d.py -c 0 -n 512 -w 1 -a 2e-4 -k 1e-12 --nu 1e-12 --T0 298 --T1 273 --name test -f 60  -v 0.4
python .\exec\rbc2d.py -c 1 -n 512 -w 1 -a 2e-5 -k 1e-12 --nu 1e-12 --T0 298 --T1 273 --name test -f 100 -v 0.1
python .\exec\rbc2d.py -c 2 -n 512 -w 1 -a 2e-4 -k 1e-12 --nu 1e-12 --T0 298 --T1 273 --name test -f 100 -v 0.1

python .\exec\rbc2d.py -c 0 -n 128 -w 1.5 -a 2.0e-5 -k 1e-6 --nu 1e-6 --T0 275.5 --T1 273 --name 1b
python .\exec\rbc2d.py -c 0 -n 128 -w 3 -a 2.0e-5 -k 1e-6 --nu 1e-6 --T0 275.5 --T1 273 --name 2b
python .\exec\rbc2d.py -c 0 -n 128 -w 4 -a 2.0e-5 -k 1e-6 --nu 1e-6 --T0 275.5 --T1 273 --name 3b

python .\exec\rbc2d.py -c 1 -n 128 -w 1.5 -a 2.0e-5 -k 1e-6 --nu 1e-6 --T0 275.5 --T1 273 --name 1b
python .\exec\rbc2d.py -c 1 -n 128 -w 3 -a 2.0e-5 -k 1e-6 --nu 1e-6 --T0 275.5 --T1 273 --name 2b
python .\exec\rbc2d.py -c 1 -n 128 -w 4 -a 2.0e-5 -k 1e-6 --nu 1e-6 --T0 275.5 --T1 273 --name 3b

python .\exec\rbc2d.py -c 2 -n 128 -w 1.5 -a 2.0e-5 -k 1e-6 --nu 1e-6 --T0 275.5 --T1 273 --name 1b
python .\exec\rbc2d.py -c 2 -n 128 -w 3 -a 2.0e-5 -k 1e-6 --nu 1e-6 --T0 275.5 --T1 273 --name 2b
python .\exec\rbc2d.py -c 2 -n 128 -w 4 -a 2.0e-5 -k 1e-6 --nu 1e-6 --T0 275.5 --T1 273 --name 3b