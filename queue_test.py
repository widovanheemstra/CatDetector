import tensorflow as tf


# Make a queue of file names including all the JPEG images files in the relative
# image directory.
file_names = tf.train.match_filenames_once(
    '/Users/widovanheemstra/Virtualenvs/python35/Projects/CatsAndDogs/data/test_labeled/*/*.jpg')
# file_names = [
#    '/Users/widovanheemstra/Virtualenvs/python35/Projects/CatsAndDogs/data/test_labeled/cats/cat.2000.jpg',
#    '/Users/widovanheemstra/Virtualenvs/python35/Projects/CatsAndDogs/data/test_labeled/cats/cat.2001.jpg']

filename_queue = tf.train.string_input_producer(file_names)

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
file_name, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)

# Fix
# image = tf.image.resize_images(image_orig, [224, 224])
# image.set_shape((224, 224, 3))
# batch_size = 50
# num_preprocess_threads = 1
# min_queue_examples = 256
# images = tf.train.shuffle_batch(
#     [image],
#     batch_size=batch_size,
#     num_threads=num_preprocess_threads,
#     capacity=min_queue_examples + 3 * batch_size,
#     min_after_dequeue=min_queue_examples)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run, use the local_variables_initializer()
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    print(image_tensor)
    # img_lst = sess.run([file_name, image_file])
    # print(img_lst)
    # img_lst = sess.run([file_name, image_file])
    # print(img_lst)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
