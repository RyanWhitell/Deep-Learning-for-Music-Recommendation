# Deep Learning for Music Recommendation 
Music streaming services use recommendation systems to improve the customer
experience by generating favorable playlists and by fostering the discovery of new music.
State of the art recommendation systems use both collaborative filtering and
content-based recommendation methods. Collaborative filtering suffers from the cold
start problem; it can only make recommendations for music for which it has enough
user data, so content-based methods are preferred. Most current content-based
recommendation systems use convolutional neural networks on the spectrograms of
track audio. The architectures are commonly borrowed directly from the field of
computer vision. It is shown in this study that musically-motivated convolutional
neural network architectures outperform architectures that are highly-optimized for
image-related tasks. A content-based recommendation model is built using
musically-motivated deep learning architectures. The model is shown to be able to map
an artist onto an artist embedding space where its nearest neighbors by cosine similarity
are related artists and make good recommendations. It is also shown that metadata,
such as lyrics, artist origin, and year, significantly improve these mappings when
combined with raw audio data.