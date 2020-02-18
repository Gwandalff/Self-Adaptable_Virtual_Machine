package visitor;

public interface IDynamicModule {
	
	public enum Strategy{
		BEFORE, AFTER, AROUND
	}
	
	boolean updateBefore(AcceptInterface self);
	void updateAfter(AcceptInterface self);
	int[] targetedNodes();
	Strategy callStrategy();
}



/*
 * @around(execution("VisitorInterface.*(..)"))
 * public void glue(ProceedingJoinPoint joinPoint){
 * 		EObject node = joinPoint.getASTNode();
 * 		node.notifyBefore();
 * 		boolean doTheMethod = node.notifyAroundStart();
 * 		if(doTheMethod){
 * 			joinPoint.proceed();
 * 		}
 * 		node.notifyAroundEnd();
 * 		node.notifyAfter();
 * }
 * 
 * */
