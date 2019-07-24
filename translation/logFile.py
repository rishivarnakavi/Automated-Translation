import logging
import os
import sys
from datetime import date
from time import gmtime, strftime

currentDirectory = os.path.dirname(__file__)

def timeIzNow():
    full = strftime("-%d-%h-%M-%S", gmtime())
    return full

fileName = "logInfo"
logName = os.path.join(currentDirectory, "logInformation/" + fileName + timeIzNow() + ".log")

open(logName, 'w').close()

logging.basicConfig(
    filename=logName,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)

def handleException(message):
    logging.error(message)


def handleInfo(message):
    logging.info(message)
    pass
