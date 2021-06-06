

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

runRemoteSageMaker:
	python3 src/part_06_runRemoteSageMakerS3/runRemoteSageMakerS3.py

batchInference:
	python3 src/part_07_batchInference/utils/createFolderStructure.py
	python3 src/part_07_batchInference/batchInference.py 
	
hpo:
	python3 src/part_08_hop/hpo.py

transferLearning:
	python3 src/part_09_startFromPrevModel/startFromPrevModel.py