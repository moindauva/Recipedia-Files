import 'dart:async';
import 'package:cloud_firestore/cloud_firestore.dart';

class FavoriteRecipeModel {
  Future<String?> addFavoriteRecipe({
    required String userID,
    required List recipeID,
  }) async {
    try {
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('favoriteRecipes');
      //Call the recipe's CollectionReference to add a new user
      /*final snapshot = await recipes.doc(userID).get();
      final data = snapshot.data() as Map<String, dynamic>;
      print(data);*/
      await recipes
          .doc(userID.toString())
          .update({'recipeID': FieldValue.arrayUnion(recipeID)});
      return 'success';
    } catch (e) {
      return 'Error adding user';
    }
  }

  Future<String?> updateFavoriteRecipe({
    required String userID,
    required List data,
  }) async {
    try {
      FirebaseFirestore.instance
          .collection('favoriteRecipes')
          .doc(userID)
          .update({'recipeID': FieldValue.arrayUnion(data)});
      return 'success';
    } catch (e) {
      return 'Error adding user';
    }
  }

  Future<bool?> checkIfRecipeAlreadyInFavorite(String userID, List data) async {
    print('data: $data');
    try {
      bool AlreadyInFavorite = false;
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('favoriteRecipes');
      await recipes.where('recipeID', isEqualTo: data).get();
      print('recipes: $recipes');
      String reci = recipes.toString();
      if (data == reci) {
        print('recipes.toString(): $reci');
        return AlreadyInFavorite = true;
      }
      return AlreadyInFavorite;
    } catch (e) {
      return false;
    }
  }

  Future<List> getFavoriteRecipeData(String userID) async {
    print('inside func');
    try {
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('favoriteRecipes');
      final snapshot = await recipes.doc(userID).get();
      final data = snapshot.data() as Map<String, dynamic>;
      return data['recipeID'];
    } catch (e) {
      return [];
    }
  }

  Future<bool> checkIfFavoriteRecipeExists(String userID) async {
    try {
      var collectionReference =
          FirebaseFirestore.instance.collection('favoriteRecipes');
      var doc = await collectionReference.doc(userID).get();
      return doc.exists;
    } catch (e) {
      return false;
    }
  }

  Future<dynamic> getFavoriteRecipeDocument(String recipeID) async {
    try {
      var collectionReference =
          FirebaseFirestore.instance.collection('favoriteRecipes');
      var doc = await collectionReference.doc(recipeID).get();
      return doc;
    } catch (e) {
      rethrow;
    }
  }

  Future<int> getFavoriteRecipeCount() async {
    int count = await FirebaseFirestore.instance
        .collection('favoriteRecipes')
        .get()
        .then((value) => value.size);
    return count;
  }
}
