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
controllers["right"] = true
controllers["A"] = false
controllers["B"] = false
controllers["start"] = false
controllers["select"] = false

-- data_to_send = {}
-- data_to_send["ready_to_listen"] = false
-- data_to_send["p1_score"] = 0
-- data_to_send["horizontal_evolution"] = 0
-- data_to_send["p1_is_alive"] = 0

local current_frame = 0
local p1_score = 0
local horizontal_evolution = 0
local room = 0
local room_increment = 0
local ready_to_listen = false
local p1_alive_status = false
local initial_state = false

while not ready_to_listen do
	if	current_frame == 20 then
		controllers["start"] = true
		joypad.set(1, controllers)
	end

	if	current_frame == 25 then
		controllers["start"] = true
		joypad.set(1, controllers)
	end

	if	current_frame == 700 and not initial_state then
		initial_state = savestate.object(5)
		savestate.save(initial_state)
		ready_to_listen = true
	end

	emu.frameadvance()

	p1_score = memory.readbyte(0x07E2)
	room = memory.readbyte(0x0065)
	room_increment = memory.readbyte(0x0064)
	horizontal_evolution = room + (room_increment*255)
	p1_alive_status = memory.readbyte(0x0090)

	current_frame = current_frame + 1
end

while true do
	s, err = incomming:receive(1024)
	if s and not err then
		data_received = json.decode(s)
		emu.message(s)
		emu.message("flag 1")
		if data_received["command"] == "frameadvance" then
			joypad.set(1, controllers)
			current_frame = current_frame + 1
			emu.message(current_frame)
			emu.frameadvance()
		end

	end
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