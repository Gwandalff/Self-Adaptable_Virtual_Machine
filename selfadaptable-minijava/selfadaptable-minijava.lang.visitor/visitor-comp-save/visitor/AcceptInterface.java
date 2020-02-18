package visitor;

import java.lang.Object;

public interface AcceptInterface {
	Object accept(VisitorInterface accept);
	
	void attach(IDynamicModule dm);
	void detach(IDynamicModule dm);
	void notifyDynamicModules();
}
