import os
import logging

currentDirectory = os.path.dirname(__file__)
logName = os.path.join(currentDirectory, 'logInformation/corpus.log')

open(logName, 'w').close()

logging.basicConfig(filename=logName,
                            filemode='a',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

# class ErrorTypes:
#     FileDownloadError, IdsMismatchError, FileWriteError = range(3)

def handleException(message):
    logging.error(message)

def handleInfo(message):
    logging.info(message)
    pass
