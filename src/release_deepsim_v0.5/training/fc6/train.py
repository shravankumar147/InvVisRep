import caffe
import numpy as np
import time
import os
import sys

if len(sys.argv) == 1:
  start_snapshot = 0
else:
  start_snapshot = int(sys.argv[1])

max_iter = int(1e6) # maximum number of iterations
display_every = 50 # show losses every so many iterations
snapshot_every = 10000 # snapshot every so many iterations
snapshot_folder = 'snapshots' # where to save the snapshots (and load from)
gpu_id = 0
feat_shape = (4096,)
recogn_feat_shape = (256,6,6)
im_size = (3,227,227)
batch_size = 64
snapshot_at_iter = -1
snapshot_at_iter_file = 'snapshot_at_iter.txt'

sub_nets = ('encoder', 'generator', 'discriminator', 'data')

if not os.path.exists(snapshot_folder):
  os.makedirs(snapshot_folder)
    
#make solvers
with open ("solver_template.prototxt", "r") as myfile:
  solver_template=myfile.read()
  
for curr_net in sub_nets:
  with open("solver_%s.prototxt" % curr_net, "w") as myfile:
    myfile.write(solver_template.replace('@NET@', curr_net))                 

#initialize the nets
caffe.set_device(gpu_id)
caffe.set_mode_gpu()
encoder = caffe.AdamSolver('solver_encoder.prototxt')
generator = caffe.AdamSolver('solver_generator.prototxt')
discriminator = caffe.AdamSolver('solver_discriminator.prototxt')
data_reader = caffe.AdamSolver('solver_data.prototxt')

encoder.net.copy_from('../../trained_models/caffenet/caffenet.caffemodel')

#load from snapshot
if start_snapshot:
  curr_snapshot_folder = snapshot_folder +'/' + str(start_snapshot)
  print '\n === Starting from snapshot ' + curr_snapshot_folder + ' ===\n'
  generator_caffemodel = curr_snapshot_folder +'/' + 'generator.caffemodel'
  if os.path.isfile(generator_caffemodel):
    generator.net.copy_from(generator_caffemodel)
  else:
    raise Exception('File %s does not exist' % generator_caffemodel)
  discriminator_caffemodel = curr_snapshot_folder +'/' + 'discriminator.caffemodel'
  if os.path.isfile(discriminator_caffemodel):
    discriminator.net.copy_from(discriminator_caffemodel)
  else:
    raise Exception('File %s does not exist' % discriminator_caffemodel)

#read weights of losses
recon_loss_weight = generator.net._blob_loss_weights[generator.net._blob_names_index['recon_loss']]
feat_loss_weight = encoder.net._blob_loss_weights[encoder.net._blob_names_index['feat_recon_loss']]
discr_loss_weight = discriminator.net._blob_loss_weights[discriminator.net._blob_names_index['discr_loss']]

train_discr = True
train_gen = True

#do training
start = time.time()
for it in range(start_snapshot,max_iter):
  # read the data
  data_reader.net.forward_simple()
  # feed the data to the encoder and run it
  encoder.net.blobs['data'].data[...] = data_reader.net.blobs['data'].data
  encoder.net.blobs['feat_gt'].data[...] = np.zeros((batch_size,) + recogn_feat_shape, dtype='float32')
  encoder.net.forward_simple()
  feat_real = np.copy(encoder.net.blobs['feat_out'].data)
  recogn_feat_real = np.copy(encoder.net.blobs['recogn_feat_out'].data)
  # feed the data to the generator and run it
  generator.net.blobs['data'].data[...] = data_reader.net.blobs['data'].data
  generator.net.blobs['feat'].data[...] = encoder.net.blobs['feat_out'].data
  generator.net.forward_simple()
  generated_img = generator.net.blobs['generated'].data
  recon_loss = generator.net.blobs['recon_loss'].data
  # encode the generated image to compare its features to the features of the input image
  encoder.net.blobs['data'].data[...] = generated_img
  encoder.net.blobs['feat_gt'].data[...] = recogn_feat_real
  encoder.net.forward_simple()
  feat_recon_loss = encoder.net.blobs['feat_recon_loss'].data
  # run the discriminator on real data
  discriminator.net.blobs['data'].data[...] = data_reader.net.blobs['data'].data
  discriminator.net.blobs['label'].data[...] = np.zeros((batch_size,1,1,1), dtype='float32')
  discriminator.net.blobs['feat'].data[...] = feat_real
  discriminator.net.forward_simple()
  discr_real_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
  if train_discr:
    discriminator.increment_iter()
    discriminator.net.clear_param_diffs()
    discriminator.net.backward_simple()
  # run the discriminator on generated data
  discriminator.net.blobs['data'].data[...] = generated_img
  discriminator.net.blobs['label'].data[...] = np.ones((batch_size,1,1,1), dtype='float32')
  discriminator.net.blobs['feat'].data[...] = feat_real
  discriminator.net.forward_simple()
  discr_fake_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
  if train_discr:
    discriminator.net.backward_simple()
    discriminator.apply_update()

  # run the discriminator on generated data with opposite labels, to get the gradient for the generator
  discriminator.net.blobs['data'].data[...] = generated_img
  discriminator.net.blobs['label'].data[...] = np.zeros((batch_size,1,1,1), dtype='float32')
  discriminator.net.blobs['feat'].data[...] = feat_real
  discriminator.net.forward_simple()
  discr_fake_for_generator_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
  if train_gen:
    generator.increment_iter()
    generator.net.clear_param_diffs()
    encoder.net.backward_simple()
    discriminator.net.backward_simple()
    
    generator.net.blobs['generated'].diff[...] = encoder.net.blobs['data'].diff + discriminator.net.blobs['data'].diff
    generator.net.backward_simple()
    generator.apply_update()
   
  #display
  if it % display_every == 0:
    print "[%s] Iteration %d: %f seconds" % (time.strftime("%c"), it, time.time()-start)
    print "  recon loss: %e * %e = %f" % (recon_loss, recon_loss_weight, recon_loss*recon_loss_weight)
    print "  feat loss: %e * %e = %f" % (feat_recon_loss, feat_loss_weight, feat_recon_loss*feat_loss_weight)
    print "  discr real loss: %e * %e = %f" % (discr_real_loss, discr_loss_weight, discr_real_loss*discr_loss_weight)
    print "  discr fake loss: %e * %e = %f" % (discr_fake_loss, discr_loss_weight, discr_fake_loss*discr_loss_weight)
    print "  discr fake loss for generator: %e * %e = %f" % (discr_fake_for_generator_loss, discr_loss_weight, discr_fake_for_generator_loss*discr_loss_weight)
    start = time.time()
    if os.path.isfile(snapshot_at_iter_file):
      with open (snapshot_at_iter_file, "r") as myfile:
        snapshot_at_iter = int(myfile.read())
    
  #snapshot
  if it % snapshot_every == 0 or it == snapshot_at_iter:
    curr_snapshot_folder = snapshot_folder +'/' + str(it)
    print '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
    if not os.path.exists(curr_snapshot_folder):
      os.makedirs(curr_snapshot_folder)
    generator_caffemodel = curr_snapshot_folder + '/' + 'generator.caffemodel'
    generator.net.save(generator_caffemodel)
    discriminator_caffemodel = curr_snapshot_folder + '/' + 'discriminator.caffemodel'
    discriminator.net.save(discriminator_caffemodel)
    
  #switch optimizing discriminator and generator, so that neither of them overfits too much
  discr_loss_ratio = (discr_real_loss + discr_fake_loss) / discr_fake_for_generator_loss
  if discr_loss_ratio < 1e-1 and train_discr:    
    train_discr = False
    train_gen = True
    print "<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen)
  if discr_loss_ratio > 5e-1 and not train_discr:    
    train_discr = True
    train_gen = True
    print " <<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen)
  if discr_loss_ratio > 1e1 and train_gen:
    train_gen = False
    train_discr = True
    print "<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen)
  
