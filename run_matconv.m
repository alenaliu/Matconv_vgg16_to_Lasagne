function [ all_scores ] = run_matconv()

%Runs sample URLs in matconvnet
%Used to validate Lasagne conversion of params
urls = load('urls.mat');

urls = cellstr(urls.urls);
net = load('imagenet-vgg-verydeep-16.mat');

for n = 1:100
    try
        %Preprocesses image
        im = imread(urls{n});
        im_ = single(im);
        
        %Anti-Aliasing must be disabled as skimage does not support it
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2), 'bilinear', 'AntiAliasing', false);

        im_(:,:,1) = im_(:,:,1) - net.meta.normalization.averageImage(1) ;
        im_(:,:,2) = im_(:,:,2) - net.meta.normalization.averageImage(2) ;
        im_(:,:,3) = im_(:,:,3) - net.meta.normalization.averageImage(3) ;
        
        %Get generated probabilities
        res = vl_simplenn(net, im_);
        scores = squeeze(gather(res(end).x));

        if n > 1
            all_scores = cat(2,all_scores,scores);
        else
            all_scores = scores;
        end
        
    catch
        all_scores = cat(2,all_scores,zeros(1000,1));
    end
end

save('/home/veda/Downloads/matconv_probs.mat','all_scores');

end

