import time
import asyncio
import multiprocessing
import random
import prediction
from concurrent.futures import ProcessPoolExecutor
import os,sys
import cv2

os.chdir("./../")

ROOT_DIR = os.path.abspath(os.getcwd())
cap = cv2.VideoCapture('test.mp4')
'''
async def collecting_loop(queue):  
    while True:
        time.sleep(.2)
        val = random.random()
        print('queuing', val)
        await queue.put(val)  


async def processing_loop(queue):  
    while True:  
        val = await queue.get()
        print('received', val)
        await asyncio.sleep(.1)


async def main():  
    queue = asyncio.Queue(maxsize=10)
    await asyncio.gather(collecting_loop(queue), processing_loop(queue))


loop = asyncio.get_event_loop()   
loop.run_until_complete(main())
'''
async def collecting_loop(queue, queue1):  
    print("cl")  
    loop = asyncio.get_event_loop()
    cap = cv2.VideoCapture('./../test.mp4')
    while True:  
        _, img = await loop.run_in_executor(None, cap.read)
        await queue.put(img)
        val = await queue1.get()
        cv2.imshow('test', val)
        cv2.waitKey(1)


async def processing_loop(queue, queue1):  
    while True:  
        val = await queue.get()
        time.sleep(1)
        await queue1.put(val)




async def main():  
    queue, queue1 = asyncio.Queue(maxsize=10), asyncio.Queue(maxsize=10)
    await asyncio.gather(collecting_loop(queue, queue1), processing_loop(queue, queue1))

'''

def readimage():
	while True:
		print('reading image')

def processimage():
	while True:
		print('process image')


exe = ProcessPoolExecutor(2)
loop = asyncio.get_event_loop()
read = asyncio.ensure_future(loop.run_in_executor(exe, readimage))
processing = asyncio.ensure_future(loop.run_in_executor(exe, processimage))
loop.run_forever()

'''



#loop.run_until_complete(main())
'''
async def something():
	while True:
		#await asyncio.sleep(0.1)
		return 3

async def test():
	while True:
		print('im starting')
		result = await something()
		print('got result: ', result)

loop = asyncio.get_event_loop()   
loop.run_until_complete(test())
'''
