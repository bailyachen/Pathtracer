compile : clean
	mkdir build
	cmake -S . -B build -D OptiX_INSTALL_DIR:PATH="C:\ProgramData\NVIDIA Corporation\OptiX SDK 6.5.0"
	cmake --build build

clean:
	rm -rf build