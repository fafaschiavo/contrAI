function reset_controllers(controllers)
	controllers["up"] = false
	controllers["down"] = false
	controllers["left"] = false
	controllers["right"] = false
	controllers["A"] = false
	controllers["B"] = false
	controllers["start"] = false
	controllers["select"] = false
	return controllers
end

pcall(require, "luarocks.require")
local json = require "cjson"

local socket = require "socket"
local INPUT_IP = '127.0.0.1'
local INPUT_PORT = 53474
local OUTPUT_IP = '127.0.0.1'
local OUTPUT_PORT = 53475
local outgoing = socket.udp()
outgoing:setpeername(OUTPUT_IP, OUTPUT_PORT)
outgoing:settimeout(0)
local incomming = socket.udp()
incomming:setsockname(INPUT_IP,INPUT_PORT)
incomming:settimeout(0)

emu.softreset()

controllers = {}
controllers["up"] = false
controllers["down"] = false
controllers["left"] = false
controllers["right"] = false
controllers["A"] = false
controllers["B"] = false
controllers["start"] = false
controllers["select"] = false

data_to_send = {}
data_to_send["ready_to_listen"] = false
data_to_send["p1_score"] = 0
data_to_send["horizontal_evolution"] = 0

local current_frame = 0
local start_time = os.time()
local end_time = os.time()
local elapsed_time = os.difftime(end_time-start_time)
local elapsed_frames = 0
local frames_per_second = 0
local elapsed_time_since_last_fps = 0
local p1_score = 0
local horizontal_evolution = 0
local room = 0
local room_increment = 0
local ready_to_listen = false

while true do

	reset_controllers(controllers)

	if	current_frame == 20 then
		controllers["start"] = true
		joypad.set(1, controllers)
	end

	if	current_frame == 25 then
		controllers["start"] = true
		joypad.set(1, controllers)
	end

	if	current_frame == 700 then
		initial_state = savestate.object(5)
		savestate.save(initial_state)
		ready_to_listen = true
		data_to_send["ready_to_listen"] = true
	end


	emu.frameadvance()

	p1_score = memory.readbyte(0x07E2)
	room = memory.readbyte(0x0065)
	room_increment = memory.readbyte(0x0064)
	horizontal_evolution = room + (room_increment*255)

	current_frame = current_frame + 1

	end_time = os.time()
	elapsed_time = os.difftime(end_time - start_time)

	if elapsed_time - elapsed_time_since_last_fps >= 1 then
		frames_per_second = (current_frame - elapsed_frames)/(elapsed_time - elapsed_time_since_last_fps)
		elapsed_frames = current_frame
		elapsed_time_since_last_fps = elapsed_time
	end

	emu.message('-----------------------')
	emu.message('FPS:' .. frames_per_second)
	emu.message('Elapsed Time:' .. elapsed_time)
	emu.message('Current frame:' .. current_frame)
	emu.message('FPS:' .. frames_per_second)
	emu.message('P1 Score:' .. p1_score)
	emu.message('Horizontal Evolution:' .. horizontal_evolution)

	data_to_send["p1_score"] = p1_score
	data_to_send["horizontal_evolution"] = horizontal_evolution
	
	gui.savescreenshotas('screenshots/' .. current_frame)

	outgoing:send(json.encode(data_to_send))
end


-- emu.message('teste')
-- emu.print('teste') -- Not working properly

-- fceux contra_rom.nes --loadlua helloworld.lua

-- D – B
-- F – A
-- Enter – Start
-- S – Select
-- Keypad up – Up
-- Keypad left – Left
-- Keypad down – Down
-- Keypad right – Right

-- savestate.load(initial_state)
-- outgoing:send('Hello world')