-- print('Hi there')
-- emu.speedmode("normal") -- Set the speed of the emulator
-- emu.loadrom("contra_rom.nes")

while true do
	-- gui.text(50,50,"Hello world!");
	emu.message("Hello there")
	emu.frameadvance() -- This essentially tells FCEUX to keep running
end