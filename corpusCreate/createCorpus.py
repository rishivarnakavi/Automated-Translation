import csv
import gzip
import os
import re
import socket
import sys
import tarfile
import time
import urllib
import zipfile
from collections import OrderedDict
from io import BytesIO
from multiprocessing import Event, Pool
from os import listdir
from tempfile import mktemp
from urllib.request import urlretrieve
from xml.dom.minidom import parse, parseString

from lxml import etree

from logFile import handleException, handleInfo

REMOTE_SERVER = "www.google.com"

currentDirectory = os.path.dirname(__file__)
inputDir = os.path.join(currentDirectory, 'input100')

if not os.path.exists(os.path.join(currentDirectory, 'output100')):
    os.makedirs(currentDirectory + '/output1s00')

outputDir = os.path.join(currentDirectory, 'output100')

if not os.path.exists(os.path.join(outputDir, 'en')):
    os.makedirs(outputDir + '/en')
if not os.path.exists(os.path.join(outputDir, 'es')):
    os.makedirs(outputDir + '/es')


def parseSubtitlesXML(xmlFileContent):
    context = etree.iterparse(BytesIO(str.encode(xmlFileContent)))
    timeFrameText = ""
    sequenceDict = {}
    for action, elem in context:
        if not elem.text:
            text = "None"
        else:
            text = elem.text
        if (elem.tag == "w"):
            if (text.lower().startswith((".", "'", "!", "@", "?", ",", "%",
                                         "&", "(", ")", ";", ":", "\""))):
                timeFrameText = timeFrameText[:-1]
            timeFrameText += text + " "
        if (elem.tag == "s"):
            timeFrameText = timeFrameText[:-1]
            sequenceDict[elem.get("id")] = timeFrameText
            timeFrameText = ""
    return sequenceDict


def startDownloadProcess(completeDict, inputFileName):
    for eachId, corpusList in completeDict.items():
        try:
            filename = mktemp('english.gz')
            urllib.request.urlretrieve(
                "http://opus.lingfil.uu.se/OpenSubtitles2016/xml/" + corpusList[0],
                filename)
            fileReader = gzip.open(filename)
            xmlContent = parseString(fileReader.read()).toxml()
            englishXMLFileContent = parseSubtitlesXML(xmlContent)
            filename = mktemp('spanish.gz')
            urllib.request.urlretrieve(
                "http://opus.lingfil.uu.se/OpenSubtitles2016/xml/" + corpusList[1],
                filename)
            fileReader = gzip.open(filename)
            xmlContent = parseString(fileReader.read()).toxml()
            spanishXMLFileContent = parseSubtitlesXML(xmlContent)
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            template = "An exception of type {0} occured. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            # print(exc_type, fname, exc_tb.tb_lineno)
            message = str(exc_type) + " : " + str(fname) + " : " + str(exc_tb.tb_lineno) + " : " + str(corpusList[0]) + " : " + str(corpusList[1])
            handleException(inputFileName + " : " + message)
            while(not is_connected()):
                time.sleep(10)
            continue

        enTimeFrameIds = corpusList[2].split(',')
        esTimeFrameIds = corpusList[3].split(',')
        movieTextEnglish = eachId + "+++++ "
        movieTextSpanish = eachId + "+++++ "
        if (len(enTimeFrameIds) == len(esTimeFrameIds)):
            for index, englishIds in enumerate(enTimeFrameIds):
                spanishIds = esTimeFrameIds[index]
                for value in englishIds.split(' '):
                    movieTextEnglish += str(englishXMLFileContent.get(value))
                for value in spanishIds.split(' '):
                    movieTextSpanish += str(spanishXMLFileContent.get(value))
                movieTextEnglish += "+++++ "
                movieTextSpanish += "+++++ "

            try:
                # print (inputFileName.split('.')[0])
                # print (outputDir + "/" + inputFileName.split('.')[0] + "_en.txt")
                fileReader = open(outputDir + "/en/" + inputFileName.split(".")[0] + "_en.txt", "a", encoding="utf8")
                fileReader.write(movieTextEnglish + "\n")

                fileReader = open(outputDir + "/es/" + inputFileName.split(".")[0] + "_es.txt", "a", encoding="utf8")
                fileReader.write(movieTextSpanish + "\n")
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # handleInfo (exc_type, fname, exc_tb.tb_lineno)
                message = str(exc_type) + " : " + str(fname) + " : " + str(exc_tb.tb_lineno) + " : " + str(corpusList[0]) + " : " + str(corpusList[1])
                handleException(message)
        else:
            handleException("Corpus not found : " + str(corpusList[0] + " : " + corpusList[1]))
        fileReader.close()

        message = inputFileName + " : " + str(eachId) + " : " + str(len(corpusList[2].split(','))) + " : " + str(len(corpusList[3].split(',')))
        handleInfo(message)


def process_file(inputFile):
    inputFileName = inputFile
    inputFile = os.path.join(inputDir, inputFile)
    currentTextFile = open(inputFile)
    completeDict = OrderedDict()
    for line in currentTextFile:
        line = re.split(r'\t+|\n', line)
        del line[-1]
        spanishId = re.split(r'\.', re.split(r'\/', line[1])[3])[0]
        if (spanishId in completeDict):
            completeDict[spanishId][2] = completeDict[spanishId][2] + "," + line[2]
            completeDict[spanishId][3] = completeDict[spanishId][3] + "," + line[3]
        else:
            completeDict[spanishId] = completeDict.get(spanishId, line)
    completeDict.popitem(last=False)[0]
    completeDict.popitem(last=True)[0]
    completeDictionary = {}

    startDownloadProcess(completeDict, inputFileName)


def is_connected():
    try:
        host = socket.gethostbyname(REMOTE_SERVER)
        s = socket.create_connection((host, 80), 2)
        return True
    except:
        pass
    return False


def setup(event):
    global unpaused
    unpaused = event

if __name__ == '__main__':
    event = Event()
    pool = Pool(24, setup, (event,))
    start_time = time.time()
    results = pool.map_async(process_file, listdir(inputDir))
    event.set()   # unpause workers
    flag = False
    while (not results.ready()):
        flag = True

    print("--- %s seconds ---" % (time.time() - start_time))

    # print ("end")
