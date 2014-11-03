%This code is written by Eugenio Sebastiani and Igor Atzeni

%This script compute the training of a ClasRBM. It provides the model and a
%history of errors during the epochs. They are saved in two structures: 
%model and errors, plus a modelavg that contain the model using the average
%techniques.

%It assume that the following variable are set:
% data: the dataset, where each row is a example,
% labels: a vector of labels for each example,
% testdata: as data but referring to the test set,
% testlabels: as labels but referring to the test set,
% restart: to start the learning it must be set to 1. If it is set to 0 the
%          learning starts from the point where it was stopped.

%Setting properly the parameter it is possible to learn only in discriminative
%way (discrim = 1, alpha = 0), only in generative way (discrim = 0, 
%alpha = 1), in an hybrid configuration (discrim = 1, apha = "some value" )

if restart == 1
  
  %parameters to set
  init_gamma = 0.9;     % initial learning rate
  tau = 1.9;            % descending speed of gamma 
  alpha = 0;            % weight of the generative learning
  h_size = 800;         % number of hidden units
  init_momentum = 0.5;  % initial value of the momentum
  final_momentum = 0.9; % value of the momentum after 5 epochs
  weight_cost = 10^-4;  % parameter of weigth decay 
  max_epocs = 150;      % number of epochs before stopping
  num_batch = 100;      % the number of batch of the dataset
  valid_size = 10000;   % the size of the validation set
  
  discrim = 1;    % 1 to activate the discriminative learning, 0 otherwise
  freeenerg = 1;  % if discrim is zero but one want to compute the free energy
  
  %initializations of sizes
  x_size = size(data, 2);
  num_class = length(unique(labels));
  
  %initializations of wights and biases
  w = randInitWeights(h_size, x_size);
  u = randInitWeights(h_size, num_class);

  b_x = zeros(x_size, 1);
  b_y = zeros(num_class, 1);
  b_h = zeros(h_size, 1);
  
  %initializations of errors
  err_classification = zeros(max_epocs,1);
  err_reconstruct = zeros(max_epocs,1);
  err_free_energ = zeros(max_epocs,1);
  
  %initializations of gradients
  delta_w = 0;
  delta_u = 0;
  delta_b_x = 0;
  delta_b_y = 0;
  delta_b_h = 0;
  
  %averaging initializations
  t = 1;
  avglast = 5;
  
  %other initializations
  restart = 0;
  i_epoch = 1;
  momentum = init_momentum;
  num_epochs = max_epocs;
end;

avgstart = num_epochs - avglast;

data_valid = data(end - valid_size + 1 : end, : );
label_valid = labels(end - valid_size + 1 : end, : );

%convert y in the notetion one out of number of classes
lab = boolean(set_y(labels(1:end - valid_size,:), num_class));

%convert the dataset in batches
[batch_data, batch_labels , batch_size] = ...
  createBatches(data(1:end - valid_size,:), lab, num_batch);

for i_epoch = i_epoch:num_epochs
  sum_rec_error = 0;
  sum_class_error = 0;
  sum_tot_free_energ = 0;
  
  %update the momentum
  if  i_epoch == 5 
      momentum = final_momentum;
  end;
  
  %simulated annealing
  gamma = (init_gamma) / (1 + (i_epoch - 1) / tau);
  
  for iBatch = 1 : num_batch

    %clamping
    x = batch_data{iBatch}';
    y = batch_labels{iBatch}';
    
    %%%%% Generative Fase %%%%%
      %compute the hiden unit
      hprob = prob_h_given_xy(x, y, w, u, b_h);       

      h = sampling(hprob);

      %compute the recostruction
      xrprob = prob_x_given_h( h, w, b_x );
      yrprob = prob_y_given_h( h, u, b_y );
      xr = sampling(xrprob);
      yr = sampling_y(yrprob);
      hrprob = prob_h_given_xy( xr, yr, w, u, b_h);

      %compute the generative gradient
      %positive gradient - negative graident
      g_w_gen = (hprob * x' - hrprob * xrprob') / batch_size;
      g_u_gen = (hprob * y' - hrprob * yrprob') / batch_size;
      g_b_x_gen = (sum(x,2) - sum(xrprob,2)) / batch_size;
      g_b_y_gen = (sum(y,2) - sum(yrprob,2)) / batch_size;
      g_b_h_gen = (sum(hprob,2) - sum(hrprob,2)) / batch_size;
    
    %%%%% Discriminative Fase %%%%%
    if (discrim || freeenerg)
      %initialization of the gradients
      g_b_h_disc_acc = 0;
      g_w_disc_acc = 0;
      g_u_disc_acc = zeros(h_size,num_class);

      %ausiliary variables
      w_times_x = w * x;
      
      o_t_y = zeros(h_size, batch_size, num_class);
      neg_free_energ = zeros(num_class, batch_size);
      for iClass= 1:num_class
        o_t_y(:,:,iClass) = bsxfun(@plus, b_h + u(:,iClass), w_times_x );
        neg_free_energ(iClass,:) = b_y(iClass) + sum(log(1 + exp(o_t_y(:,:,iClass))));
      end;
      
      sum_tot_free_energ = sum_tot_free_energ + sum(-neg_free_energ(y));
      
      if discrim %if only generative but we want the free energy

        %we subtract its mean to the free energy in order to keep its values
        %smaller as possible, otherwise the exponetial in p(y|x) gets only inf
        %values.
        med = mean2(neg_free_energ);      
        p_y_given_x = exp(neg_free_energ - med);
        p_y_given_x = bsxfun(@rdivide, p_y_given_x, sum(p_y_given_x));

        dif_y_p_y = y - p_y_given_x;

        %update the gradient of the bias of y for the class 'iClass'
        g_b_y_disc_acc = sum(dif_y_p_y,2);

        %looping over classes is more efficient
        for iClass= 1:num_class
          sig_o_t_y = sig(o_t_y(:,:,iClass)); 

          sig_o_t_y_selected = sig_o_t_y(:,y(iClass,:)); 

          %update the gradient of the bias of h for the class 'iClass'
          g_b_h_disc_acc = g_b_h_disc_acc + sum(sig_o_t_y_selected, 2) - ...
            sum(bsxfun(@times, sig_o_t_y, p_y_given_x(iClass,:)),2);

          %update the gradient of w fot the class 'iClass'
          g_w_disc_acc = g_w_disc_acc + sig_o_t_y_selected * x(:,y(iClass,:))' - ...
            bsxfun(@times, sig_o_t_y, p_y_given_x(iClass,:)) * x';

          %update the gradient of u fot the class 'iClass'
          g_u_disc_acc(:,iClass) = sum(bsxfun(@times, sig_o_t_y, dif_y_p_y(iClass,:)), 2);
        end;

        %NOTE: the bias of x are not updated in the discriminative fase.

          g_b_y_disc = g_b_y_disc_acc / batch_size;
          g_b_h_disc = g_b_h_disc_acc / batch_size;
          g_w_disc = g_w_disc_acc / batch_size;
          g_u_disc = g_u_disc_acc / batch_size;
      else
        g_b_y_disc = 0;
        g_b_h_disc = 0;
        g_w_disc = 0;
      	g_u_disc = 0;
      end;
    end;
    
    %compute the recostruction and classification error
    sum_rec_error = sum_rec_error + sum(sum((x - xrprob) .^ 2));

    %update w, u, b_x, b_y and b_h
    delta_w = delta_w * momentum + ...
      gamma * (discrim * g_w_disc + alpha * g_w_gen - weight_cost * w);
    delta_u = delta_u * momentum + ...
      gamma * (discrim * g_u_disc + alpha * g_u_gen - weight_cost * u);
    delta_b_x = delta_b_x * momentum + ...
      gamma * alpha * g_b_x_gen;
    delta_b_y = delta_b_y * momentum + ...
      gamma * (discrim * g_b_y_disc + alpha * g_b_y_gen);
    delta_b_h = delta_b_h * momentum + ...
      gamma * (discrim * g_b_h_disc + alpha * g_b_h_gen);
    w = w + delta_w;
    u = u + delta_u;
    b_x = b_x + delta_b_x;
    b_y = b_y + delta_b_y;
    b_h = b_h + delta_b_h;
    
    if (i_epoch > avgstart)
    %apply averaging
      w_avg = w_avg - (1/t)*(w_avg - w);
      b_x_avg = b_x_avg - (1/t)*(b_x_avg - b_x);
      b_h_avg = b_h_avg - (1/t)*(b_h_avg - b_h);
      u_avg = u_avg - (1/t)*(u_avg - u);
      b_y_avg = b_y_avg - (1/t)*(b_y_avg - b_y);
      t = t+1;
    else
      w_avg = w;
      b_h_avg = b_h;
      b_x_avg = b_x;
      u_avg = u;
      b_y_avg = b_y;
    end
  end;
  
  %errors management 
  err_classification(i_epoch) = predict( data_valid, label_valid, ...
    num_class, b_y, b_h, w, u );
  
  err_reconstruct(i_epoch) = sum_rec_error;
  err_free_energ(i_epoch) = sum_tot_free_energ;
  
  fprintf('Epoch: %d\n\tRecon error: %f\n\tClassification error on validation: %f\n\tFree energy: %f\n\n', ...
    i_epoch, sum_rec_error, err_classification(i_epoch), sum_tot_free_energ);

  
end;

err_testset_avg = predict( testdata, testlabels, num_class, ...
  b_y_avg, b_h_avg, w_avg, u_avg );
err_testset = predict( testdata, testlabels, num_class, b_y, b_h, w, u );
fprintf('Error averaging on the test set: %f\n%', err_testset_avg);
fprintf('Error on the test set: %f\n%', err_testset_avg);

model.u = u;
model.w = w;
model.b_x = b_x;
model.b_y = b_y;
model.b_h = b_h;
modelavg.b_y = b_y_avg; 
modelavg.b_x = b_x_avg;
modelavg.b_h = b_h_avg;
modelavg.w = w_avg;
modelavg.u = u_avg;
errors.err_classification = err_classification;      
errors.err_reconstruct = err_reconstruct;
errors.err_free_energ = err_free_energ;
errors.err_testset_avg = err_testset_avg;
errors.err_testset = err_testset;