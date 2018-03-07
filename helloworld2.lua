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
local INPUT_PORT = 53477
local OUTPUT_IP = '127.0.0.1'
local OUTPUT_PORT = 53476
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
data_to_send["p1_is_alive"] = 0
data_to_send["features"] = {}

features = {}

current_frame = 0
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
local p1_alive_status = false
local initial_state = false
data_received = controllers

while true do
	if	ready_to_listen and (current_frame%2) == 0 then
		controllers["up"] = data_received["up"]
		controllers["down"] = data_received["down"]
		controllers["left"] = data_received["left"]
		controllers["right"] = data_received["right"]
		controllers["A"] = data_received["A"]
		controllers["B"] = data_received["B"]
		controllers["start"] = false
		controllers["select"] = false
		joypad.set(1, controllers)
	elseif ready_to_listen and (current_frame%2) > 0 then
		controllers["up"] = data_received["up"]
		controllers["down"] = data_received["down"]
		controllers["left"] = data_received["left"]
		controllers["right"] = data_received["right"]
		controllers["A"] = false
		controllers["B"] = false
		controllers["start"] = false
		controllers["select"] = false
		joypad.set(1, controllers)
	end

	outgoing:send(json.encode(data_to_send))
	s, err = incomming:receive(4096)
	if s and not err then
		controllers = reset_controllers(controllers)
		data_received = json.decode(s)
		controllers["up"] = data_received["up"]
		controllers["down"] = data_received["down"]
		controllers["left"] = data_received["left"]
		controllers["right"] = data_received["right"]
		if (current_frame%2) == 0 then
			controllers["A"] = data_received["A"]
			controllers["B"] = data_received["B"]
		else
			controllers["A"] = false
			controllers["B"] = false
		end
		controllers["start"] = data_received["start"]
		controllers["select"] = data_received["select"]
		joypad.set(1, controllers)
		if data_received["soft_reset"] then
			emu.softreset()
			current_frame = 0
			savestate.load(initial_state)
		end
	end

	if	current_frame == 20 then
		controllers["start"] = true
		joypad.set(1, controllers)
	end

	if	current_frame == 25 then
		controllers["start"] = true
		joypad.set(1, controllers)
	end

	-- if	current_frame >= 650 and current_frame < 800 and (current_frame%50) > 0 then
	-- 	controllers["right"] = true
	-- 	joypad.set(1, controllers)
	-- end

	if	current_frame == 700 and not initial_state then
		controllers["start"] = false
		controllers["right"] = false
		initial_state = savestate.object(5)
		savestate.save(initial_state)
		ready_to_listen = true
		data_to_send["ready_to_listen"] = true
	end


	emu.frameadvance()

	p1_score = memory.readbyte(0x07E2)
	room = memory.readbyte(0x0065)
	room_increment = memory.readbyte(0x0064)

	p1_alive_status = memory.readbyte(0x0090)
	horizontal_evolution = room + (room_increment*255)

	features["p1_alive_status"] = p1_alive_status
	features["horizontal_evolution"] = horizontal_evolution
	features["weapon"] = memory.readbyte(0x00AA)
	features["invulnerability_timer"] = memory.readbyte(0x00AE)
	features["invencibility_timer"] = memory.readbyte(0x00B0)
	features["water_status"] = memory.readbyte(0x00B2)
	features["y_delta"] = memory.readbyte(0x00C6)
	features["x_position"] = memory.readbyte(0x031A)
	features["y_position"] = memory.readbyte(0x0334)
	features["b0x"] = memory.readbyte(0x03B8)
	features["b1x"] = memory.readbyte(0x03B9)
	features["b2x"] = memory.readbyte(0x03BA)
	features["b3x"] = memory.readbyte(0x03BB)
	features["b4x"] = memory.readbyte(0x03BC)
	features["b5x"] = memory.readbyte(0x03BD)
	features["b6x"] = memory.readbyte(0x03BE)
	features["b7x"] = memory.readbyte(0x03BF)
	features["b8x"] = memory.readbyte(0x03C0)
	features["b9x"] = memory.readbyte(0x03C1)
	features["b0y"] = memory.readbyte(0x03C8)
	features["b1y"] = memory.readbyte(0x03C9)
	features["b2y"] = memory.readbyte(0x03CA)
	features["b3y"] = memory.readbyte(0x03CB)
	features["b4y"] = memory.readbyte(0x03CC)
	features["b5y"] = memory.readbyte(0x03CD)
	features["b6y"] = memory.readbyte(0x03CE)
	features["b7y"] = memory.readbyte(0x03CF)
	features["b8y"] = memory.readbyte(0x03D0)
	features["b9y"] = memory.readbyte(0x03D1)

	turret_delay = memory.readbyterange(0x0538, 0x000F)
	for i=1,15 do
		current_index = 'td' .. i
		features[current_index] = string.byte(turret_delay, i)
	end

	enemy_shooting_delay = memory.readbyterange(0x0567, 0x000F)
	for i=1,15 do
		current_index = 'esd' .. i
		features[current_index] = string.byte(enemy_shooting_delay, i)
	end

	enemy_work = memory.readbyterange(0x0568, 0x000F)
	for i=1,15 do
		current_index = 'ew' .. i
		features[current_index] = string.byte(enemy_work, i)
	end

	enemy_hit_points = memory.readbyterange(0x0578, 0x000F)
	for i=1,15 do
		current_index = 'ehp' .. i
		features[current_index] = string.byte(enemy_hit_points, i)
	end

	platforms = memory.readbyterange(0x0698, 0x005F)
	for i=1,95 do
		current_index = 'p' .. i
		features[current_index] = string.byte(platforms, i)
	end

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
	-- emu.message('P1 Score:' .. p1_score)
	-- emu.message('Horizontal Evolution:' .. horizontal_evolution)
	-- emu.message('Alive Status:' .. p1_alive_status)

	data_to_send["features"] = features
	data_to_send["p1_score"] = p1_score
	data_to_send["p1_is_alive"] = p1_alive_status
	data_to_send["horizontal_evolution"] = horizontal_evolution
	
	-- gui.savescreenshotas('screenshots/' .. current_frame)
	data_to_send["last_frame"] = current_frame

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