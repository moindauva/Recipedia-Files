class FavoriteRecipes {
  String recipeID;
  String recipeName;
  String recipeDescription;
  String recipeURL;
  String recipeRating;
  String recipeTime;
  List<String> recipeIngredients = [];

  FavoriteRecipes({
    required this.recipeID,
    required this.recipeName,
    required this.recipeDescription,
    required this.recipeURL,
    required this.recipeRating,
    required this.recipeTime,
    required this.recipeIngredients,
  });
}
