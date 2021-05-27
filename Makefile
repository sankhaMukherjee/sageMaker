

generateData:
	python3 'src/part_01_genertaeData/generateData.py'
	ls -lh data

runLocal:
	python3 src/part_02_runLocal/runLocal.py

runLocalArgs:
	python3 src/part_03_runLocalArgs/runLocalArgs.py

runLocalSageMaker:
	python3 src/part_04_runLocalSageMaker/runLocalSageMaker.py