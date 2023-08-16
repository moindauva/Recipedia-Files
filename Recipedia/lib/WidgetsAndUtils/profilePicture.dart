import 'dart:io';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart';

class profilePicture {
  // Upload the profile picture to Firebase Storage
  /*Future<String> uploadProfilePicture(File file) async {
    // Initialize Firebase Storage
    FirebaseStorage storage = FirebaseStorage.instance;

    // Upload the file to Firebase Storage
    Reference ref = storage.ref().child('profile_pictures/${DateTime.now().millisecondsSinceEpoch}');
    UploadTask uploadTask = ref.putFile(file);
    TaskSnapshot taskSnapshot = await uploadTask;

    // Get the download URL of the file
    String downloadURL = await taskSnapshot.ref.getDownloadURL();

    return downloadURL;
  }*/

  Future<String> uploadAndSaveProfilePicture(
      File imageFile, String userId) async {
    /*final Reference storageRef = FirebaseStorage.instance.ref().child('images');
    final UploadTask uploadTask = storageRef.child('image.jpg').putFile(imageFile);

    final TaskSnapshot taskSnapshot = await uploadTask.whenComplete(() => null);
    final String imageUrl = await taskSnapshot.ref.getDownloadURL();

    DocumentReference userRef =  FirebaseFirestore.instance.collection('users').doc(userId);
    userRef.update({'profile_picture': imageUrl});*/
    final fileName = basename(imageFile.path);
    final destination = 'images/$fileName';
    final storage = FirebaseStorage.instance;

    try {
      await storage.ref(destination).putFile(imageFile);
      final url = await storage.ref(destination).getDownloadURL();
      DocumentReference userRef =
          FirebaseFirestore.instance.collection('users').doc(userId);
      userRef.update({'profilePictureURL': url});
      return url;
    } on FirebaseException catch (e) {
      return 'https://static.vecteezy.com/system/resources/thumbnails/005/544/718/small/profile-icon-design-free-vector.jpg';
    }
  }

  Future<String> getProfilePicture(String userId) async {
    FirebaseFirestore firestore = FirebaseFirestore.instance;
    DocumentSnapshot userSnapshot =
        await firestore.collection('users').doc(userId).get();
    String downloadURL = userSnapshot.get('profilePictureURL');

    return downloadURL;
  }

  // Load the user's profile picture from the download URL
  Widget buildProfilePicture(String downloadURL) {
    return CachedNetworkImage(
      imageUrl: downloadURL,
      placeholder: (context, url) => const CircularProgressIndicator(),
      errorWidget: (context, url, error) => const Icon(Icons.error),
    );
  }
}
