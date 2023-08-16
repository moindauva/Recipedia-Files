import 'package:cloud_firestore/cloud_firestore.dart';
import '../Admin/recipes/recipes.dart';

class RecipeModel {
  Future<String?> addRecipe({
    required int recipeID,
    required String recipeName,
    required String recipeDescription,
    required String recipeIngredients,
    required String recipeTime,
    required String recipeImageURL,
    required String recipeDiet,
    required String recipeCourse,
    required String recipeCategory,
    required String recipeRating,
  }) async {
    try {
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('recipes');
      //Call the recipe's CollectionReference to add a new user
      await recipes.doc(recipeID.toString()).set({
        'recipe_name': recipeName,
        'recipe_description': recipeDescription,
        'recipe_ingredients': recipeIngredients,
        'recipe_rating': recipeRating,
        'recipe_time': recipeTime,
        'recipeImageURL': recipeImageURL,
        'recipe_category': recipeCategory,
        'recipe_course': recipeCourse,
        'recipe_diet': recipeDiet,
      });
      return 'success';
    } catch (e) {
      return 'Error adding recipe';
    }
  }

  Future<String?> UpdateRecipe({
    required int recipeID,
    required String recipeName,
    required String recipeDescription,
    required String recipeIngredients,
    required String recipeTime,
    required String recipeImageURL,
    required String recipeDiet,
    required String recipeCourse,
    required String recipeCategory,
    required String recipeRating,
  }) async {
    try {
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('recipes');
      //Call the recipe's CollectionReference to add a new user
      await recipes.doc(recipeID.toString()).update({
        'recipe_name': recipeName,
        'recipe_description': recipeDescription,
        'recipe_ingredients': recipeIngredients,
        'recipe_rating': recipeRating,
        'recipe_time': recipeTime,
        'recipeImageURL': recipeImageURL,
        'recipe_category': recipeCategory,
        'recipe_course': recipeCourse,
        'recipe_diet': recipeDiet,
      });
      return 'success';
    } catch (e) {
      return 'Error updating recipe';
    }
  }

  Future<String?> getRecipeData(String recipeID, String key) async {
    try {
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('recipes');
      final snapshot = await recipes.doc(recipeID).get();
      final data = snapshot.data() as Map<String, dynamic>;
      return data[key];
    } catch (e) {
      return 'Error fetching data';
    }
  }

  Future<List> getRecipeingredients(String recipeID) async {
    try {
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('recipes');
      final snapshot = await recipes.doc(recipeID).get();
      final data = snapshot.data() as Map<String, dynamic>;
      return data['recipe_ingredients'];
    } catch (e) {
      return [];
    }
  }

  Future<List> getRecipeDataList() async {
    List recipes = [];
    int count = await getRecipeCount();
    int recipeID = 101;
    for (int i = 0; i < count; i++, recipeID++) {
      if (await checkIfRecipeDocExists(recipeID.toString()) == true) {
        QuerySnapshot<Map<String, dynamic>> recipesSnapshot =
            await FirebaseFirestore.instance.collection('recipes').get();
        for (QueryDocumentSnapshot<Map<String, dynamic>> doc
            in recipesSnapshot.docs) {
          final data = doc.data();
          var recipe = Recipes(
            recipeID: doc.id,
            recipeName: data['recipe_name'],
            recipeCategory: data['recipe_category'],
            recipeDescription: data['recipe_description'],
            recipeURL: data['recipeImageURL'],
            recipeRating: data['recipe_rating'],
            recipeTime: data['recipe_time'],
            recipeIngredients:
                (data['recipe_ingredients'] as List<dynamic>).cast<String>(),
          );
          recipes.add(recipe);
        }
      }
    }
    return recipes;
  }

  /*Future<List<String>> getRecipeIngredients(String recipeId)async{
    List<String> recipeIngredients= [];
    QuerySnapshot<Map<String,dynamic>> query = await FirebaseFirestore.instance.collection('recipes').where("id", isEqualTo: recipeId).get();
    if(query.docs.isNotEmpty){
      LinkedHashMap<String,dynamic> map = query.docs[0].data()['recipe_ingredients'];
      for(var rec in map.values){
        recipeIngredients.add(rec);
      }
    }
    return recipeIngredients;
  }*/

  Future<String?> deleteRecipeData(String recipeID) async {
    try {
      CollectionReference recipes =
          FirebaseFirestore.instance.collection('recipes');
      await recipes.doc(recipeID).delete();
      return 'Recipe deleted';
    } catch (e) {
      return 'Try Again';
    }
  }

  Future<bool?> checkIfRecipeExists(String recipeName) async {
    try {
      bool recipeExists = false;
      await FirebaseFirestore.instance
          .collection('recipes')
          .where('recipe_name', isEqualTo: recipeName)
          .get()
          .then((value) {
        recipeExists = true;
      });
      return recipeExists;
    } catch (e) {
      return false;
    }
  }

  Future<dynamic> getRecipeDocument(String recipeID) async {
    try {
      var collectionReference =
          FirebaseFirestore.instance.collection('recipes');
      var doc = await collectionReference.doc(recipeID).get();
      return doc;
    } catch (e) {
      rethrow;
    }
  }

  Future<int> getRecipeCount() async {
    int count = await FirebaseFirestore.instance
        .collection('recipes')
        .get()
        .then((value) => value.size);
    return count;
  }

  Future<bool> checkIfRecipeDocExists(String recipeID) async {
    try {
      var collectionReference =
          FirebaseFirestore.instance.collection('recipes');
      var doc = await collectionReference.doc(recipeID).get();
      return doc.exists;
    } catch (e) {
      rethrow;
    }
  }
}
