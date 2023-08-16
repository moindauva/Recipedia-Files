import 'package:cloud_firestore/cloud_firestore.dart';

class RatingModel {
  Future<List<Map<String, Map<String, dynamic>>>> feedbackList() async {
    List<Map<String, Map<String, dynamic>>> commentList = [];
    List<String> recipeIDs = await geRecipeIdsList();
    CollectionReference recipes =
        FirebaseFirestore.instance.collection('feedbacks');
    for (int i = 0; i < recipeIDs.length; i++) {
      final snapshot = await recipes.doc(recipeIDs[i]).get();
      final data = snapshot.data() as Map<String, dynamic>;
      Map<String, dynamic> transformedData = {};
      transformedData[recipeIDs[i]] = data;
      commentList.add(transformedData.cast<String, Map<String, dynamic>>());
    }
    return commentList;
  }

  Future<List<String>> geRecipeIdsList() async {
    List<String> recipeIDs = [];
    QuerySnapshot snapshot =
        await FirebaseFirestore.instance.collection('feedbacks').get();
    snapshot.docs.forEach((doc) {
      recipeIDs.add(doc.id);
    });
    return recipeIDs;
  }

  /*Future<String> isUserIDExists(String recipeID, String key) async {
    try {
      CollectionReference recipes =
      FirebaseFirestore.instance.collection('feedbacks');
      final snapshot = await recipes.doc(recipeID).get();
      final data = snapshot.data() as Map<String, dynamic>;
      return data[key];
    } catch (e) {
      return 'Error fetching data';
    }
  }*/

  Future<List<String>> isUserIDExistsInFeedback(String recipeID) async {
    print(recipeID);
    try {
      /* CollectionReference recipes =
      FirebaseFirestore.instance.collection('feedbacks');
      final snapshot = await recipes.doc(recipeID).get();
      final data = snapshot.data() as Map<String, dynamic>;
      return data['userID'];*/

      final List<String> data = (await FirebaseFirestore.instance
              .collection('feedbacks')
              .doc(recipeID)
              .get())
          .data()!['userID']
          .cast<String>();
      return data;
    } catch (e) {
      return ['Custom', 'list'];
    }
  }

  Future<String> updateRating(String recipeID, String comment, String userName,
      String userID, int rating) async {
    try {
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('feedbacks');
      await recipes.doc(recipeID).update({
        'comment': FieldValue.arrayUnion([comment]),
        'ratings': FieldValue.arrayUnion([rating]),
        'userID': FieldValue.arrayUnion([userID]),
        'username': FieldValue.arrayUnion([userName]),
      });
      return 'success';
    } catch (e) {
      return 'error';
    }
  }

  Future<String> addRating(String recipeID, String comment, String userName,
      String userID, int rating) async {
    try {
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('feedbacks');
      await recipes.doc(recipeID).set({
        'comment': FieldValue.arrayUnion([comment]),
        'ratings': FieldValue.arrayUnion([rating]),
        'userID': FieldValue.arrayUnion([userID]),
        'username': FieldValue.arrayUnion([userName]),
      });
      return 'success';
    } catch (e) {
      return 'error';
    }
  }

  Future<bool> checkIfFeedbackExists(String recipeID) async {
    try {
      var collectionReference =
          FirebaseFirestore.instance.collection('feedbacks');
      var doc = await collectionReference.doc(recipeID).get();
      return doc.exists;
    } catch (e) {
      rethrow;
    }
  }
}
