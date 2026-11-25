# Builds libface_align.so, the nvdspreprocess custom library that aligns faces for ArcFace.
# The kernel is compiled by nvcc, the DeepStream glue by g++ (nvcc cannot parse the glib headers).
CUDA_VER ?= 12.6
CUDA := /usr/local/cuda-$(CUDA_VER)
DS := /opt/nvidia/deepstream/deepstream
CXXFLAGS := -std=c++17 -O3 -fPIC -I$(DS)/sources/includes             -I$(DS)/sources/gst-plugins/gst-nvdspreprocess/include -I$(CUDA)/include             $(shell pkg-config --cflags gstreamer-1.0 glib-2.0)
LIBS := -L$(DS)/lib -lnvbufsurftransform -lnvbufsurface -L$(CUDA)/lib64 -lcudart

libface_align.so: face_align.o face_align_kernel.o
	g++ -shared -o $@ $^ $(LIBS)

face_align.o: face_align.cpp face_align_kernel.h
	g++ $(CXXFLAGS) -c -o $@ $<

face_align_kernel.o: face_align_kernel.cu face_align_kernel.h
	$(CUDA)/bin/nvcc -O3 -Xcompiler -fPIC -c -o $@ $<

clean:
	rm -f *.o libface_align.so
