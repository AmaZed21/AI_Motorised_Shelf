import asyncio
import csv
from datetime import datetime
import random
import numpy as np

try:
    import vosk
    import sounddevice as sd
    import queue
    import json
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

#Defining directions
DIR_NONE = "NONE"
DIR_UP   = "UP"
DIR_DOWN = "DOWN"

#Defining states
STATE_STOPPED     = "STOPPED"
STATE_MOVING_UP   = "MOVING_UP"
STATE_MOVING_DOWN = "MOVING_DOWN"

#Features of compartment
class Compartment:
    #Defaults
    MAX_SPEED = 7.0
    MIN_SPEED = 0.0
    MIN_HEIGHT = 0.0
    MAX_HEIGHT = 60.0
    MAX_WEIGHT = 1.0
    SENSOR_THRESHOLD = 2.0

    def __init__(self, com_no: int = 1, weight: float = 0.0, contents: list = []) -> None:
        self.com_no = com_no #store compartment number
        self.contents = contents #items inside the compartment
        self.position = 60.0 #Starting position (stowed)
        self.speed = self.MIN_SPEED #Stopped
        self.state = STATE_STOPPED
        self.direction = DIR_NONE
        self.weight = weight if contents is not None else 0.0
        self.sensor_distance = 62.0

    def move_up(self) -> None:
        if (self.position < self.MAX_HEIGHT) and (self.state in (STATE_STOPPED, STATE_MOVING_DOWN)):
            self.direction = DIR_UP
            self.state = STATE_MOVING_UP
    
    def move_down(self) -> None:
        if (self.position > self.MIN_HEIGHT) and (self.state in (STATE_STOPPED, STATE_MOVING_UP)):
            self.direction = DIR_DOWN
            self.state = STATE_MOVING_DOWN

    def stop(self) -> None:
        if self.state != STATE_STOPPED:
            self.state = STATE_STOPPED
            self.direction = DIR_NONE
            self.speed = self.MIN_SPEED
        else:
            print('Already stopped')
    
    def update(self, dt: float = 0.1) -> None: 
        #Speed control when moving
        if self.state in (STATE_MOVING_DOWN, STATE_MOVING_UP):
            if self.MAX_SPEED > self.speed:
                self.speed += 0.25
                
        #Calculate height when travelling upwards
        if self.state == STATE_MOVING_UP:
            self.position += self.speed * dt
            self.sensor_distance += self.speed * dt
            if (self.position >= self.MAX_HEIGHT - 1):
                self.position = self.MAX_HEIGHT
                self.stop()

        #Calculate height when travelling downwards
        if self.state == STATE_MOVING_DOWN:
            self.position -= self.speed * dt
            self.sensor_distance -= self.speed * dt
            if (self.position <= self.MIN_HEIGHT + 1):
                self.position = self.MIN_HEIGHT
                self.stop()

        #Stop when weight exceeded
        if (self.weight > self.MAX_WEIGHT) and (self.state in (STATE_MOVING_DOWN, STATE_MOVING_UP)):
            self.stop()
    
        #stop if obstruction detected (ultrasonic sensor used)
        if self.state == STATE_MOVING_DOWN and self.sensor_distance < self.SENSOR_THRESHOLD:
            self.stop()
            print('Obstruction detected, emergency stop')
    
    #Retrieve status of compartment
    def print_status(self) -> None:
        if len(self.contents) > 0:
            print(f'''Compartment no. {self.com_no}
Items: {', '.join(self.contents)}
Weight: {self.weight: .2f} kg
Height from ground: {self.position: .0f} cm
Speed: {self.speed: .2f} cm/s
State: {self.state}
Direction: {self.direction}
    ''')
        else:
            print(f'''Compartment no. {self.com_no}
Items: Empty
Weight: NA
Height from ground: {self.position: .0f} cm
Speed: {self.speed: .2f} cm/s
State: {self.state}
Direction: {self.direction}
    ''')

#Features of shelf
class Shelf:
    #Keep track of all compartments in the shelf
    def __init__(self, total_com: list = []) -> None:
        self.total_com = total_com
    
    #Refresh status of all compartments in shelf
    def update_all(self, dt: float = 0.1) -> None:
        for comp in self.total_com:
            comp.update(dt)
        
    #Emergency stop
    def emergency_stop(self):
        for comp in self.total_com:
            comp.stop()
        print('Emergency stop activated, all operations halted')
    
    #Status retrieval
    def get_status(self):
        for comp in self.total_com:
            comp.print_status()
    
    #Status of specific compartment
    def get_spec_stat(self, comp_no: int):
        com = next((c for c in self.total_com if c.com_no == comp_no), None)
        if com:
            com.print_status()
        else:
            print('Invalid compartment no.')
    
    def reset(self):
        for c in self.total_com:
            c.move_up()
            c.sensor_distance = 62.0
        print('Successfully reset')
    
    def find_item(self, item):
        try:
            item = item.lower().strip()

            for com in self.total_com:
                normalized_contents = [x.lower().strip() for x in com.contents]

                if item in normalized_contents:
                    return com
        except AttributeError:
            return

#Event logging
class Logger:
    def __init__(self, filename: str = 'logs.csv'):
        self.filename = filename
        self.fields = ['timestamp', 'compartment_no', 'items', 'position', 'state', 'event_type']

        #Create csv file
        with open(self.filename, 'w', newline = '') as f:
            writer = csv.DictWriter(f, fieldnames = self.fields)
            writer.writeheader()
    
    def log(self, compartment, event_type: str = 'TICK'):
        with open(self.filename, 'a', newline = '') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'compartment_no': compartment.com_no,
                'items': ', '.join(compartment.contents) if compartment.contents else 'Empty',
                'position': compartment.position,
                'state': compartment.state,
                'event_type': event_type
            })

#Voice commands
class Voice:
    def __init__(self, shelf, model_path, device=None, samplerate=None):
        if VOICE_AVAILABLE:
            self.shelf = shelf
            self.device = device
            self.audio_queue = queue.Queue()

            if samplerate is None:
                device_info = sd.query_devices(device, "input")
                samplerate = int(device_info["default_samplerate"]) #type: ignore

            self.samplerate = samplerate
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.samplerate)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.audio_queue.put(bytes(indata))

    def normalize_text(self, text):
        return text.lower().strip()

    def extract_item(self, text):
        text = self.normalize_text(text)
        if text == 'stop all' or text == 'stop':
            return ('stop', 'all')
        elif text.startswith('stop '):
            item = text[len('stop '):].strip()
            if item:
                return ('stop', item)
        elif text.startswith('bring '):
            item = text[len('bring '):].strip()
            if item:
                return ('bring', item)
        elif text.startswith('retract '):
            item = text[len('retract '):].strip()
            if item:
                return ('retract', item)
        return None

    def handle_command(self, text):
        text = self.normalize_text(text)
        if not text:
            return
        print(f"Detected speech: {text}")
        result = self.extract_item(text)

        if result is None:
            return

        action, item = result

        if action == 'stop':
            if item is None:
                self.shelf.emergency_stop()
                print("All movement stopped.")
            else:
                com = self.shelf.find_item(item)
                if com:
                    com.stop()
                    print(f"Stopped compartment with {item}.")
            return

        if action == 'bring':
            com = self.shelf.find_item(item)
            if com:
                com.move_down()
                print(f"Bringing {item}.")
            return
        
        if action == 'retract':
            if item == 'empty':
                for com in self.shelf.total_com:
                    if not com.contents:
                        com.move_up()
            print("Retracting empty compartments.")
        else:
            comp = self.shelf.find_item(item)
            if comp:
                comp.move_up()
                print(f"Retracting {item}.")

    
    def listen_loop(self):
        if VOICE_AVAILABLE:
            with sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=8000,
                device=self.device,
                dtype="int16",
                channels=1,
                callback=self.audio_callback
            ):
                while True:
                    data = self.audio_queue.get()
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "")
                        self.handle_command(text)

#Execute commands
async def process_command(shelf, com, command, logger):
    if command == 'up':
            com.move_up()
    elif command == 'down':
        com.move_down()
    elif command == 'stop':
        com.stop()
    elif command == 'reset':
        shelf.reset()
    elif command == 'block':
        com.sensor_distance = 0.5
    elif command == 'free':
        com.sensor_distance = 62.0
        if com.position != 60.0:
            com.move_down()
    else:
        print('Invalid command')
        return 

    logger.log(com, event_type=f'COMMAND_{command.upper()}')
    shelf.get_status()

#Refresh comp_information every (ticks) seconds
async def run_simulation(shelf, logger, ticks:float = 0.1):
    #initial print
    shelf.get_status()
    was_moving = {c.com_no: False for c in shelf.total_com}
    
    while True:
        shelf.update_all(ticks)

        # Only print if any compartment is moving
        if any(c.state != STATE_STOPPED for c in shelf.total_com) or any(was_moving[c.com_no] and c.state == STATE_STOPPED 
                                                                         for c in shelf.total_com):
            shelf.get_status()

        #Log events
        for c in shelf.total_com:
            if c.state != STATE_STOPPED:
                logger.log(c, event_type='TICK')
            elif was_moving[c.com_no] and c.state == STATE_STOPPED:
                logger.log(c, event_type='STOPPED')

        for c in shelf.total_com:
            was_moving[c.com_no] = c.state != STATE_STOPPED
        
        await asyncio.sleep(ticks)

#Manual cycles
async def manual_cycle(shelf, logger):
    while True:
        #Command
        raw = await asyncio.get_running_loop().run_in_executor(None, input, "Command: ")

        try:
            com_str, command = raw.split(',')
            com_num = int(com_str.strip())
            command = command.strip().lower()
        except ValueError:
            print("Please use com_num, command (1, down/up/stop/reset/block)")
            continue
        
        #Get comp
        com = next((c for c in shelf.total_com if c.com_no == com_num), None)
        if com is None:
                print("Invalid compartment number")
                continue
        
        await process_command(shelf, com, command, logger)

#Automating cycles
async def auto_cycles(shelf, logger, cycles=50):
    commands = ['up', 'down', 'stop', 'block', 'free', 'reset']
    weights  = [0.30, 0.30, 0.05, 0.15, 0.10, 0.10  ]

    for i in range(cycles):
        command = random.choices(commands, weights=weights, k=1)[0]
        com = random.choice(shelf.total_com)

        await process_command(shelf, com, command, logger)

        if random.random() < 0.5:
            while any(c.state != STATE_STOPPED for c in shelf.total_com):
                await asyncio.sleep(1)

async def main():
    logger = Logger('data/logs.csv')
    
    #Creating compartments
    com_1 = Compartment()
    com_2 = Compartment(2, weight = 0.2, contents=['towel'])
    com_3 = Compartment(3, weight = 0.5, contents = ['inhaler', 'medicine'])
    
    #Creating shelf
    shelf_1 = Shelf([com_1, com_2, com_3])

    voice = Voice(shelf_1, model_path = 'models/vosk-model-small-en-us-0.15')

    loop = asyncio.get_running_loop()

    await asyncio.gather(
            run_simulation(shelf_1, logger),
            manual_cycle(shelf_1, logger),
            loop.run_in_executor(None, voice.listen_loop)
        )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Power cut, program exited')
