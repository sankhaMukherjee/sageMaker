

generateData:
	python3 'src/part_01_genertaeData/generateData.py'
	ls -lh data

runLocal:
	python3 src/part_02_runLocal/runLocal.py

runLocalArgs:
	python3 src/part_03_runLocalArgs/runLocalArgs.py

runLocalSageMaker:
	python3 src/part_04_runLocalSageMaker/runLocalSageMaker.py

runLocalSageMakerS3:
	python3 src/part_05_runLocalSageMakerS3/runLocalSageMakerS3.py