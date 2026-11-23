function [rate, pred] = saumya_nn(trainProj, trainLabels, testProj, testLabels)
% SAUMYA_NN  Nearest neighbour with squared Euclidean distance.
%
%   rate   percentage of test columns classified correctly
%   pred   the predicted label of every test column
%
%   There is no training and nothing to converge: the "model" IS the training
%   set.  All the learning happened when we chose the projection.
%
%   We compare SQUARED distances and never take a square root, because sqrt
%   is increasing and so cannot change which neighbour is nearest.

pred = zeros(1, size(testProj, 2));
for i = 1:size(testProj, 2)
    % trainProj - testProj(:,i) subtracts the test column from EVERY training
    % column at once; sum(...,1) collapses each to one number.
    distances = sum((trainProj - testProj(:, i)).^2, 1);
    [~, best] = min(distances);
    pred(i) = trainLabels(best);
end
rate = 100 * sum(pred == testLabels) / numel(testLabels);
end
