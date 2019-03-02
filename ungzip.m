clear; close all; clc;

%newGuy = gunzip('./train-images-idx3-ubyte.gz');
fid = fopen('train-images-idx3-ubyte', 'r', 'b');

%%
header = fread(fid, 1, 'int32');
count = fread(fid, 1, 'int32');

%%
h = fread(fid, 1, 'int32');
w = fread(fid, 1, 'int32');

imgs = zeros([h w 60000]);

for i=1:count
        for y=1:h
            imgs(y,:,i) = fread(fid, w, 'uint8');
        end
end

fclose(fid);



%%
lbls = fopen('train-labels-idx1-ubyte', 'r', 'b');


%%
labels = cell(60000,1);
count = fread(lbls, 1);
for i=1:count
    labels(i,:) = fread(lbls, w );
end

