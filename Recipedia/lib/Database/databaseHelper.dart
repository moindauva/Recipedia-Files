import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:recipedia/WidgetsAndUtils/favorite_recipe_model.dart';
import 'package:sqflite/sqflite.dart';

class DatabaseHelper {
  final String favoriteRecipeTable = 'favoriteRecipe';
  final String favoriteRecipeDataTable = 'favoriteRecipeDataTable';
  final String userID = 'userID';
  final String recipeIDs = 'recipeIDs';
  final String recipeTable = 'recipes';
  final String recipeId = 'recipeId';
  final String recipeName = 'recipeName';
  final String recipeCategory = 'recipeCategory';
  final String recipeDescription = 'recipeDescription';
  final String recipeRating = 'recipeRating';
  final String recipeCourse = 'recipeCourse';
  final String recipeDiet = 'recipeDiet';
  final String recipeTime = 'recipeTime';
  final String recipeIngredients = 'recipeIngredients';
  final String recipeURL = 'recipeURL';

  static final DatabaseHelper _instance = DatabaseHelper._internal();

  factory DatabaseHelper() => _instance;

  DatabaseHelper._internal();

  Database? _db;

  Future<Database?> get db async {
    if (_db != null) {
      return _db!;
    }
    _db = await _initDatabase();
    return _db!;
  }

  Future<Database> _initDatabase() async {
    final String path = await getDatabasesPath();
    return openDatabase(
      '$path/recipes.db',
      version: 1,
      onCreate: (db, version) {
        db.execute(
            'CREATE TABLE $recipeTable($recipeId INTEGER PRIMARY KEY, $recipeName TEXT, $recipeDescription TEXT, $recipeCategory TEXT, $recipeDiet TEXT, $recipeCourse TEXT, $recipeIngredients TEXT, $recipeURL TEXT, $recipeTime TEXT, $recipeRating INTEGER)');
        db.execute(
            'CREATE TABLE $favoriteRecipeDataTable($recipeId INTEGER PRIMARY KEY, $recipeName TEXT, $recipeDescription TEXT, $recipeCategory TEXT, $recipeDiet TEXT, $recipeCourse TEXT, $recipeIngredients TEXT, $recipeURL TEXT, $recipeTime TEXT, $recipeRating INTEGER)');
        db.execute(
            'CREATE TABLE $favoriteRecipeTable($userID INTEGER PRIMARY KEY, $recipeIDs TEXT)');
      },
    );
  }

  Future<void> syncDataFromFirestore() async {
    final QuerySnapshot<Map<String, dynamic>> snapshot =
        await FirebaseFirestore.instance.collection('recipes').get();
    var dbClient = await db;
    final Batch batch = dbClient!.batch();
    for (final QueryDocumentSnapshot<Map<String, dynamic>> doc
        in snapshot.docs) {
      final Map<String, dynamic> data = doc.data();
      //List list = data['recipe_ingredients'];
      //String commaSeparatedIngredients = list.join(', ');
      batch.insert(
        conflictAlgorithm: ConflictAlgorithm.replace,
        recipeTable,
        {
          recipeId: doc.id,
          recipeName: data['recipe_name'],
          recipeCategory: data['recipe_category'],
          recipeDescription: data['recipe_description'],
          recipeRating: data['recipe_rating'],
          recipeCourse: data['recipe_course'],
          recipeDiet: data['recipe_diet'],
          recipeTime: data['recipe_time'],
          recipeIngredients: data['recipe_ingredients'],
          recipeURL: data['recipeImageURL'],
        },
      );
    }
    await batch.commit();
  }

  /*Future<void> syncDataFromFirestore() async {
    final QuerySnapshot<Map<String, dynamic>> snapshot =
        await FirebaseFirestore.instance.collection('recipes').get();
    var dbClient = await db;
    final Batch batch = dbClient!.batch();
    for (final QueryDocumentSnapshot<Map<String, dynamic>> doc
        in snapshot.docs) {
      final Map<String, dynamic> data = doc.data();
      List list = data['recipe_ingredients'];
      String commaSeparatedIngredients = list.join(', ');
      batch.insert(
        conflictAlgorithm: ConflictAlgorithm.replace,
        recipeTable,
        {
          recipeId: doc.id,
          recipeName: data['recipe_name'],
          recipeCategory: data['recipe_category'],
          recipeDescription: data['recipe_description'],
          recipeRating: data['recipe_rating'],
          recipeTime: data['recipe_time'],
          recipeIngredients: commaSeparatedIngredients,
          recipeURL: data['recipeImageURL'],
        },
      );
    }
    await batch.commit();
  }*/

  Future<void> syncFavoriteRecipesDataFromFirestore(String userID) async {
    //String? email = await SharedPreference().getCred('email');
    //String? userID = await UserModel().getUserID(email!);
    var dbClient = await db;

    bool docExist =
        await FavoriteRecipeModel().checkIfFavoriteRecipeExists(userID);
    if (docExist == false) {
      final CollectionReference collectionReference =
          FirebaseFirestore.instance.collection('favoriteRecipes');
      final String documentId = userID;
      await collectionReference.doc(documentId).set(<String, dynamic>{});
    }
    var collectionReference =
        FirebaseFirestore.instance.collection('favoriteRecipes');
    var doc = await collectionReference.doc(userID).get();
    doc = await collectionReference.doc(userID).get();
    final Map<String, dynamic>? data = doc.data();
    String id = doc.id;
    if (data!['recipeID'] != null) {
      List list = data['recipeID'];
      String commaSeparatedRecipes = list.join(', ');
      await dbClient!.insert(
        favoriteRecipeTable,
        {'userID': id, recipeIDs: commaSeparatedRecipes},
        conflictAlgorithm: ConflictAlgorithm.replace,
      );
    } else {
      await dbClient!.insert(
        favoriteRecipeTable,
        {'userID': id},
        conflictAlgorithm: ConflictAlgorithm.replace,
      );
    }
    List<int> recipesID = await getFavoriteRecipeIDs(int.parse(id));
    for (int i = 0; i < recipesID.length; i++) {
      Map<String, dynamic>? row = await getRecipe(recipesID[i]);
      await dbClient.insert(
        favoriteRecipeDataTable,
        row!,
        conflictAlgorithm: ConflictAlgorithm.replace,
      );
    }
  }

  /*Future<void> syncDataToFirestore() async {
    final List<Map<String, dynamic>> rows =
    await _db.query('my_table', orderBy: 'id ASC');

    for (final Map<String, dynamic> row in rows) {
      await FirebaseFirestore.instance
          .collection('my_collection')
          .doc(row['id'].toString())
          .set({
        'id': row['id'],
        'name': row['name'],
        'age': row['age'],
      });
    }
  }*/

  Future<void> syncData(String userID) async {
    await syncDataFromFirestore();
    await syncFavoriteRecipesDataFromFirestore(userID);
  }

  Future<String> saveRecipe(
    int recipesId,
    String recipesName,
    String recipesDescription,
    String recipesCategory,
    String recipesDiet,
    String recipesCourse,
    String recipesIngredients,
    String recipesURL,
    String recipesTime,
    int recipesRating,
  ) async {
    try {
      var dbClient = await db;
      await dbClient!.rawQuery(
          'INSERT INTO $recipeTable ($recipeId, $recipeName, $recipeDescription, $recipeCategory, $recipeCourse, $recipeDiet, $recipeIngredients, $recipeURL, $recipeTime, $recipeRating) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
          [
            recipesId,
            recipesName,
            recipesDescription,
            recipesCategory,
            recipesDiet,
            recipesCourse,
            recipesIngredients,
            recipesURL,
            recipesTime,
            recipesRating,
          ]);
      return 'Success';
    } catch (e) {
      return 'Failure';
    }
  }

  Future<List<int>> getAllRecipeID() async {
    var dbClient = await db;
    List<Map<String, dynamic>> recipeIDList =
        await dbClient!.rawQuery('SELECT $recipeId FROM $recipeTable');

    return List.generate(recipeIDList.length, (i) {
      return (recipeIDList[i][recipeId]);
    });
  }

  Future<List<Map<String, dynamic>>> random5Recipes() async {
    var dbClient = await db;
    List<Map<String, dynamic>> list = await dbClient!.rawQuery(
        'SELECT * FROM $recipeTable WHERE $recipeRating = 5 ORDER BY RANDOM() LIMIT 5');
    return list;
  }

  Future<List<Map<String, dynamic>>> getRecipeByIngredient(List ingredients) async {
    var dbClient = await db;
    String query = 'SELECT * FROM recipes WHERE ';

    for (int i = 0; i < ingredients.length; i++) {
      query += "$recipeIngredients LIKE '%${ingredients[i]}%'";

      if (i < ingredients.length - 1) {
        query += ' AND ';
      }
    }
    List<Map<String, dynamic>> list = await dbClient!.rawQuery(query);
    return list;
  }

  Future<List<Map<String, dynamic>>> getAllRecipe(
      String tableName, String categoryValue) async {
    var dbClient = await db;
    if (categoryValue == 'Appetizer') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: [categoryValue]);
      return list;
    } else if (categoryValue == 'Main Course') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: [categoryValue]);
      return list;
    } else if (categoryValue == 'Breakfast') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: [categoryValue]);
      return list;
    } else if (categoryValue == 'Lunch') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: [categoryValue]);
      return list;
    } else if (categoryValue == 'Dinner') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: [categoryValue]);
      return list;
    } else if (categoryValue == 'Side Dish') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: [categoryValue]);
      return list;
    } else if (categoryValue == 'Dessert') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: ['Desesrt']);
      return list;
    } else if (categoryValue == 'Bruntch') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: [categoryValue]);
      return list;
    } else if (categoryValue == 'Snack') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: [categoryValue]);
      return list;
    } else if (categoryValue == 'One Pot') {
      List<Map<String, dynamic>> list = await dbClient!.query(recipeTable,
          where: '$recipeCourse = ?', whereArgs: ['One Pot Dish']);
      return list;
    } else {
      List<Map<String, dynamic>> list =
          await dbClient!.rawQuery('SELECT * FROM $tableName');
      return list;
    }
  }

  /*Future<List<Map<String, dynamic>>> getRecipe(int id) async {
    var dbClient = await db;
    List<Map<String, dynamic>> list = await dbClient!
        .rawQuery('SELECT * FROM $recipeTable WHERE $recipeId = $id');
    return list;
  }*/

  Future<Map<String, dynamic>?> getRecipe(int id) async {
    var dbClient = await db;

    List<Map<String, dynamic>> row = await dbClient!.query(
      recipeTable,
      where: '$recipeId = ?',
      whereArgs: [id],
      limit: 1,
    );

    if (row.isNotEmpty) {
      return row.first;
    } else {
      return null;
    }
  }

  Future<int> updateRecipe(Map<String, dynamic> recipe) async {
    var dbClient = await db;
    int res = await dbClient!.update(recipeTable, recipe,
        where: '$recipeId = ?', whereArgs: [recipe[recipeId]]);
    return res;
  }

  Future<int> deleteRecipe(int id) async {
    var dbClient = await db;
    int res = await dbClient!
        .delete(recipeTable, where: '$recipeId = ?', whereArgs: [id]);
    return res;
  }

  Future<List<int>> getFavoriteRecipeIDs(int id) async {
    var dbClient = await db;
    List<Map<String, dynamic>> favoriteRecipeIDs = await dbClient!.query(
      favoriteRecipeTable,
      columns: [recipeIDs],
      where: '$userID = ?',
      whereArgs: [id],
    );
    if (favoriteRecipeIDs.isEmpty || favoriteRecipeIDs[0][recipeIDs] == null) {
      return [];
    } else {
      List<dynamic> dynamicList = favoriteRecipeIDs[0][recipeIDs]
          .split(',')
          .map((e) => e.trim())
          .toList();
      List<String> temp =
          dynamicList.map((element) => element.toString()).toList();
      List<int> favoriteRecipeList = [];
      for (String str in temp) {
        favoriteRecipeList.add(int.parse(str));
      }
      return favoriteRecipeList;
    }
  }

  Future<void> saveFavoriteRecipes(int id, int recipeId) async {
    var dbClient = await db;
    List<int> recipeID = await getFavoriteRecipeIDs(id);
    recipeID.add(recipeId);
    String favoriteRecipeIDs = recipeID.join(', ');
    await dbClient!.execute(
        'UPDATE $favoriteRecipeTable SET $recipeIDs = ? WHERE $userID = ?',
        [favoriteRecipeIDs, id]);
  }

  Future<bool> checkIfRecipeExists(int id, int recipeId) async {
    List<int>? recipeID = await getFavoriteRecipeIDs(id);
    for (int i = 0; i < recipeID.length; i++) {
      if (recipeId == recipeID[i]) {
        return true;
      }
    }
    return false;
  }
}
