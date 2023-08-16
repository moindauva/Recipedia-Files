/*import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:sqflite/sqflite.dart';
import 'dart:async';
import 'package:path/path.dart';

class DatabaseSync {
  final String recipeTable = 'recipeTable';
  final String recipeId = 'recipeId';
  final String recipeName = 'recipeName';
  final String recipeCategory = 'recipeCategory';
  final String recipeDescription = 'recipeDescription';
  final String recipeRating = 'recipeRating';
  final String recipeTime = 'recipeTime';
  final String recipeIngredients = 'recipeIngredients';
  final String recipeURL = 'recipeURL';

  Future<void> syncFromCloudFirestoreToSQLite() async {
    final FirebaseFirestore firestore = FirebaseFirestore.instance;
    final Database database = await openDatabase(
        join(await getDatabasesPath(), 'recipes.db'),
        version: 1, onCreate: (db, version) {
      // Create the table in the SQLite database.
      db.execute('CREATE TABLE $recipeTable('
          '$recipeId INTEGER PRIMARY KEY, '
          '$recipeName TEXT, '
          '$recipeDescription TEXT, '
          '$recipeCategory TEXT, '
          '$recipeIngredients TEXT, '
          '$recipeURL TEXT, '
          '$recipeTime TEXT, '
          '$recipeRating INTEGER)');
    });

    // Retrieve data from Cloud Firestore.
    final QuerySnapshot snapshot = await firestore.collection('recipes').get();

    // Insert the data into the SQLite database.
    final Batch batch = database.batch();
    for (var doc in snapshot.docs) {
      batch.insert(recipeTable, {
        recipeId: doc.id,
        recipeName: doc['recipe_name'],
        recipeDescription: doc['recipe_description'],
        recipeCategory: doc['recipe_category'],
        recipeIngredients:
            (doc['recipe_ingredients'] as List<dynamic>).cast<String>(),
        recipeURL: doc['recipeImageURL'],
        recipeTime: doc['recipe_time'],
        recipeRating: doc['recipe_rating'],
      });
    }
    await batch.commit();
  }

  // Function to listen to changes in Cloud Firestore and update the SQLite database.
  Future<StreamSubscription<QuerySnapshot<Object?>>>
      syncFromCloudFirestoreToSQLiteOnChange() async {
    final FirebaseFirestore firestore = FirebaseFirestore.instance;
    final Database database = await openDatabase(
        join(await getDatabasesPath(), 'recipes.db'),
        version: 1, onCreate: (db, version) {
      // Create the table in the SQLite database.
      db.execute('CREATE TABLE $recipeTable('
          '$recipeId INTEGER PRIMARY KEY, '
          '$recipeName TEXT, '
          '$recipeDescription TEXT, '
          '$recipeCategory TEXT, '
          '$recipeIngredients TEXT, '
          '$recipeURL TEXT, '
          '$recipeTime TEXT, '
          '$recipeRating INTEGER)');
    });

    // Listen to changes in Cloud Firestore.
    final StreamSubscription<QuerySnapshot> subscription = firestore
        .collection('recipes')
        .snapshots()
        .listen((QuerySnapshot snapshot) {
      final Batch batch = database.batch();
      for (var doc in snapshot.docs) {
        batch.insert(
            recipeTable,
            {
              recipeId: doc.id,
              recipeName: doc['recipe_name'],
              recipeDescription: doc['recipe_description'],
              recipeCategory: doc['recipe_category'],
              recipeIngredients:
                  (doc['recipe_ingredients'] as List<dynamic>).cast<String>(),
              recipeURL: doc['recipeImageURL'],
              recipeTime: doc['recipe_time'],
              recipeRating: doc['recipe_rating'],
            },
            conflictAlgorithm: ConflictAlgorithm.replace);
      }
      batch.commit();
    });
    return subscription;
  }
}*/

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:sqflite/sqflite.dart';

class SyncService {
  static const String usersCollection = 'recipes';
  static const String recipeId = 'recipeId';
  static const String recipeName = 'recipeName';
  static const String recipeCategory = 'recipeCategory';
  static const String recipeDescription = 'recipeDescription';
  static const String recipeRating = 'recipeRating';
  static const String recipeTime = 'recipeTime';
  static const String recipeIngredients = 'recipeIngredients';
  static const String recipeURL = 'recipeURL';

  final FirebaseFirestore firestore = FirebaseFirestore.instance;
  final Database sqfliteDb;

  SyncService(this.sqfliteDb);

  Future<void> syncData() async {
    // Fetch data from Cloud Firestore
    final QuerySnapshot<Map<String, dynamic>> snapshot =
    await firestore.collection(usersCollection).get();

    // Insert data into Sqflite database
    final Batch batch = sqfliteDb.batch();
    for (final QueryDocumentSnapshot<Map<String, dynamic>> doc
    in snapshot.docs) {
      final Map<String, dynamic> data = doc.data();
      batch.insert(
        usersCollection,
        {
          recipeId: doc.id,
          recipeName: data['recipe_name'],
          recipeCategory: data['recipe_category'],
          recipeDescription: data['recipe_description'],
          recipeRating: data['recipeRating'],
          recipeTime: data['recipeTime'],
          recipeIngredients: data['recipeIngredients'],
          recipeURL: data['recipeURL'],
        },
      );
    }
    await batch.commit();

    // Set up listener for changes in Cloud Firestore
    firestore.collection(usersCollection).snapshots().listen((snapshot) {
      final Batch batch = sqfliteDb.batch();
      for (final QueryDocumentSnapshot<Map<String, dynamic>> doc
      in snapshot.docs) {
        final Map<String, dynamic> data = doc.data();
        batch.update(
          usersCollection,
          {
            recipeId: doc.id,
            recipeName: data['recipe_name'],
            recipeCategory: data['recipe_category'],
            recipeDescription: data['recipe_description'],
            recipeRating: data['recipeRating'],
            recipeTime: data['recipeTime'],
            recipeIngredients: data['recipeIngredients'],
            recipeURL: data['recipeURL'],
          },
          where: '$recipeId = ?',
          whereArgs: [doc.id],
        );
      }
      batch.commit();
    });
  }
}

