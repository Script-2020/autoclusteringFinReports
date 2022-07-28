import time
import datetime
import math
import logging

class Logger():
    def __init__(self):
        self. start_time = time.time()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        print('Starting ' + str(datetime.datetime.now()))

    @staticmethod
    def printLog(*messages, no_time = False, logging_level='INFO'):
        message = Logger.unwrapMessage(messages)
        if no_time:
            print(message)
        else:
            print(str(datetime.datetime.now()) + '\t' + message)
    
    @staticmethod
    def unwrapMessage(*messages):
        message = ''
        for m in messages[0]:
            message += str(m) + ' '
        return message
    
    
    def getElapsedTime(self):
        time_min, str_report  = self.calculateElapsedTime()
        print(str_report)
        return time_min

    def calculateElapsedTime(self):
        totalSeconds = time.time() - self.start_time
        hours = math.floor(totalSeconds / 3600)
        minutes = math.floor(totalSeconds / 60 - hours * 60)
        seconds = totalSeconds - (hours * 3600 + minutes * 60)

        endDate = datetime.datetime.now()
        str_report = 'Time: ' + str(endDate)
        str_report += '\n' + "--- Total Time: %s hours: %s minutes  %s seconds " % (str(hours), str(minutes), str(seconds))

        time_min = int((hours * 60 + minutes + seconds /60)*100)/100
        return time_min, str_report
