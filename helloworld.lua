
local current_frame = 0
local start_time = os.time()
local end_time = os.time()
local elapsed_time = os.difftime(end_time-start_time)
local elapsed_frames = 0
local frames_per_second = 0

while true do

	emu.frameadvance()
	current_frame = current_frame + 1

	if	current_frame == 500 then
		emu.softreset()
	end

	end_time = os.time()
	elapsed_time = os.difftime(end_time - start_time)

	if (elapsed_time % 1) == 0 then
		frames_per_second = elapsed_frames - current_frame
		elapsed_frames = current_frame
	end

	emu.message('-----------------------')
	emu.message('Current frame:' .. current_frame)
	emu.message('Elapsed Time:' .. elapsed_time)
	emu.message('FPS:' .. frames_per_second)
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