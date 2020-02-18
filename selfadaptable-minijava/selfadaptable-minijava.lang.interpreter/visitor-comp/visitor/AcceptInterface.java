package visitor;

public interface AcceptInterface {
	Object accept(VisitorInterface accept);
	
	void attach(IDynamicModule dm);
	void detach(IDynamicModule dm);
	boolean notifyDynamicModulesBefore();
	void notifyDynamicModulesAfter();
}
